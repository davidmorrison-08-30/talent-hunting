"""The in-process drip dispatcher.

Approved messages can be handed to this queue instead of being sent inside the
approval request. The dispatcher drains the queue sequentially with the same
randomized pacing delay, so twenty approved messages never burst out in a few
seconds.

The queue itself lives in Postgres (the ``APPROVED`` status), so a crash does
not lose work: :func:`resume` picks up where the process left off.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import drafts as draft_agent
from app.config import Settings, get_settings
from app.db import get_sessionmaker
from app.models import OutreachRecord, OutreachStatus
from app.services import pacing
from app.x.client import XClient

log = structlog.get_logger(__name__)

#: How long the dispatcher sleeps between records.
IDLE_SLEEP_SECONDS = 5.0


class DripDispatcher:
    """Sends approved outreach records one at a time with pacing delays."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._wake = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, settings: Settings | None = None) -> None:
        """Start the background drain loop."""
        if self.running:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run())
        log.info("dispatcher_started")

    async def stop(self) -> None:
        """Ask the dispatcher to finish its current record and exit."""
        if self._task is None:
            return
        self._stopping.set()
        self._wake.set()
        try:
            await asyncio.wait_for(self._task, timeout=120)
        except (TimeoutError, asyncio.CancelledError):  # pragma: no cover
            self._task.cancel()
        self._task = None
        log.info("dispatcher_stopped")

    def notify(self) -> None:
        """Wake the dispatcher so it picks up newly queued records."""
        self._wake.set()

    async def _run(self) -> None:
        settings = get_settings()
        while not self._stopping.is_set():
            try:
                sent = await self._drain_once(settings)
            except Exception as exc:  # noqa: BLE001
                log.error("dispatcher_error", error=str(exc))
                sent = 0

            if sent == 0:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=IDLE_SLEEP_SECONDS)
                except TimeoutError:
                    pass
                finally:
                    self._wake.clear()

    async def _drain_once(self, settings: Settings) -> int:
        """Send every currently approved record, one at a time."""
        maker = get_sessionmaker()
        sent = 0

        async with maker() as session:
            result = await session.execute(
                select(OutreachRecord)
                .where(OutreachRecord.status == OutreachStatus.APPROVED.value)
                .order_by(OutreachRecord.created_at.asc())
                .limit(10)
            )
            records = list(result.scalars().all())

        for record in records:
            if self._stopping.is_set():
                break
            async with maker() as session:
                fresh = await session.get(OutreachRecord, record.id)
                if fresh is None or fresh.status != OutreachStatus.APPROVED.value:
                    continue
                try:
                    await pacing.assert_daily_cap_available(session)
                except Exception as exc:  # noqa: BLE001
                    log.info("dispatcher_cap_reached", error=str(exc))
                    break

                async with XClient(session) as client:
                    outcome = await draft_agent.approve_and_send(
                        session,
                        client,
                        record=fresh,
                        settings=settings,
                    )
                sent += 1
                log.info(
                    "dispatcher_sent",
                    record_id=str(outcome.record_id),
                    status=outcome.status,
                )
        return sent


#: The process-wide dispatcher.
dispatcher = DripDispatcher()


async def enqueue(session: AsyncSession, record_id: uuid.UUID) -> None:
    """Mark a record as queued and wake the dispatcher."""
    record = await session.get(OutreachRecord, record_id)
    if record is not None:
        record.status = OutreachStatus.APPROVED.value
        await session.flush()
    dispatcher.notify()


async def resume(settings: Settings | None = None) -> int:
    """Restart the dispatcher and send anything left in the APPROVED state."""
    dispatcher.start(settings)
    return await dispatcher._drain_once(settings or get_settings())
