"""Background execution of a discovery run.

A discovery run waits on the X API and on several LLM inferences, which takes
far longer than an API Gateway integration is allowed to hold a connection.
The request therefore only creates the run and hands the work to this module,
and the caller polls the run's status while it progresses.

The run is stored in Postgres, so the work is not lost when the process
restarts: :func:`resume_pending` picks up runs that were still marked as
running.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.search import SearchLimits, run_discovery
from app.config import Settings, get_settings
from app.db import get_sessionmaker
from app.models import DiscoveryRun
from app.x.client import XClient

log = structlog.get_logger(__name__)

#: The status a run carries while it is being worked on.
RUNNING = "RUNNING"
#: The status a run carries when it finished successfully.
COMPLETED = "COMPLETED"
#: The status a run carries when it could not be completed.
FAILED = "FAILED"

#: The statuses that mean a run is still in flight.
ACTIVE_STATUSES = (RUNNING, "PENDING")


async def start_run(
    session: AsyncSession,
    *,
    target_segment: list[str],
    seed_accounts: list[str],
    limits: SearchLimits,
    settings: Settings | None = None,
    include_bots: bool = False,
    bot_score_threshold: float = 0.6,
    known_human_accounts: list[str] | None = None,
) -> uuid.UUID:
    """Create the run row and schedule the work.

    The row is written before the work starts so the caller can poll it right
    away, and the search itself purges the previous history when it begins.
    """
    settings = settings or get_settings()

    run = DiscoveryRun(
        input_type=_input_type(target_segment, seed_accounts),
        target_segment=list(target_segment or []),
        seed_accounts=list(seed_accounts or []),
        known_human_accounts=list(known_human_accounts or []),
        status=RUNNING,
        stats={},
    )
    session.add(run)
    await session.flush()
    await session.commit()

    asyncio.create_task(
        _execute(
            run_id=run.id,
            target_segment=list(target_segment or []),
            seed_accounts=list(seed_accounts or []),
            limits=limits,
            settings=settings,
            include_bots=include_bots,
            bot_score_threshold=bot_score_threshold,
            known_human_accounts=list(known_human_accounts or []),
        )
    )
    log.info("discovery_run_scheduled", run_id=str(run.id))
    return run.id


async def _execute(
    *,
    run_id: uuid.UUID,
    target_segment: list[str],
    seed_accounts: list[str],
    limits: SearchLimits,
    settings: Settings,
    include_bots: bool,
    bot_score_threshold: float,
    known_human_accounts: list[str] | None = None,
) -> None:
    """Run the discovery in the background and record the outcome."""
    maker = get_sessionmaker()
    try:
        async with maker() as session:
            async with XClient(session) as client:
                result = await run_discovery(
                    session,
                    client,
                    run_id=run_id,
                    target_segment=target_segment,
                    seed_accounts=seed_accounts,
                    limits=limits,
                    settings=settings,
                    include_bots=include_bots,
                    bot_score_threshold=bot_score_threshold,
                    known_human_accounts=known_human_accounts or [],
                )
            await session.commit()
            log.info(
                "discovery_run_completed",
                run_id=str(result.run_id),
                returned=len(result.accounts),
                candidates=result.candidates_seen,
            )
    except Exception as exc:  # noqa: BLE001 - the run must record its own failure
        log.error("discovery_run_failed", run_id=str(run_id), error=str(exc))
        await _mark_failed(maker, run_id, str(exc))


async def _mark_failed(
    maker: Any, run_id: uuid.UUID, error: str
) -> None:
    """Record the failure on the run so the poller can stop waiting."""
    try:
        async with maker() as session:
            run = await session.get(DiscoveryRun, run_id)
            if run is not None:
                run.status = FAILED
                run.stats = {"error": error}
                await session.flush()
                await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.error("discovery_run_mark_failed_error", run_id=str(run_id), error=str(exc))


async def resume_pending(settings: Settings | None = None) -> int:
    """Re-run the runs that were still in flight when the process stopped."""
    maker = get_sessionmaker()
    settings = settings or get_settings()

    async with maker() as session:
        result = await session.execute(
            select(DiscoveryRun).where(DiscoveryRun.status.in_(ACTIVE_STATUSES))
        )
        pending = list(result.scalars().all())

    for run in pending:
        limits = SearchLimits(
            max_accounts=settings.discovery_max_accounts,
            max_candidates=settings.discovery_max_candidates,
            max_posts_per_account=settings.discovery_max_posts_per_account,
            spend_ceiling_usd=settings.discovery_spend_ceiling_usd,
            post_read_cost_usd=settings.post_read_cost_usd,
            user_read_cost_usd=settings.user_read_cost_usd,
        )
        asyncio.create_task(
            _execute(
                run_id=run.id,
                target_segment=list(run.target_segment or []),
                seed_accounts=list(run.seed_accounts or []),
                limits=limits,
                settings=settings,
                include_bots=False,
                bot_score_threshold=0.6,
                # The accounts the operator vouched for are remembered on the
                # run row, so a resumed run keeps them.
                known_human_accounts=list(run.known_human_accounts or []),
            )
        )
    if pending:
        log.info("discovery_runs_resumed", count=len(pending))
    return len(pending)


def _input_type(target_segment: list[str] | None, seed_accounts: list[str] | None) -> str:
    if target_segment and seed_accounts:
        return "both"
    if seed_accounts:
        return "seed_accounts"
    return "target_segment"
