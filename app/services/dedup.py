"""The hard dedup ledger.

``build-docs/x-rules-and-limits.md`` requires that the system "strictly track
outreach state and never double-message anyone". This module is the single
place where that invariant is enforced, together with the permanent opt-out
suppression list.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import DuplicateOutreachError, SuppressedError
from app.models import OutreachRecord, OutreachStatus, SuppressionEntry

log = structlog.get_logger(__name__)

#: Statuses that block any further outreach to the same recipient.
BLOCKING_STATUSES = (
    OutreachStatus.PENDING_APPROVAL.value,
    OutreachStatus.SENT.value,
    OutreachStatus.OPTED_OUT.value,
)


async def find_active_record(
    session: AsyncSession, x_user_id: str, *, exclude_id: uuid.UUID | None = None
) -> OutreachRecord | None:
    """Return the live outreach record for a recipient, if one exists.

    ``exclude_id`` lets the approval step re-run the dedup check while
    ignoring the very record that is being approved — otherwise the TOCTOU
    guard would always trip over its own record.
    """
    stmt = select(OutreachRecord).where(
        OutreachRecord.x_user_id == x_user_id,
        OutreachRecord.status.in_(BLOCKING_STATUSES),
    )
    if exclude_id is not None:
        stmt = stmt.where(OutreachRecord.id != exclude_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def is_suppressed(session: AsyncSession, x_user_id: str) -> bool:
    """True when the recipient is on the permanent opt-out list."""
    result = await session.execute(
        select(SuppressionEntry).where(SuppressionEntry.x_user_id == x_user_id)
    )
    return result.scalar_one_or_none() is not None


async def assert_can_contact(
    session: AsyncSession, x_user_id: str, *, exclude_id: uuid.UUID | None = None
) -> None:
    """Raise if the recipient may not be contacted.

    Checks, in order:

    1. the permanent suppression list (an explicit opt-out always wins),
    2. an existing live outreach record (the hard dedup lock).

    ``exclude_id`` skips a specific record during the approval-time
    re-check so a record does not block its own send.
    """
    if await is_suppressed(session, x_user_id):
        raise SuppressedError(
            "This account opted out and must not be contacted",
            details={"x_user_id": x_user_id},
        )

    existing = await find_active_record(session, x_user_id, exclude_id=exclude_id)
    if existing is not None:
        raise DuplicateOutreachError(
            "An outreach record already exists for this account",
            details={
                "x_user_id": x_user_id,
                "record_id": str(existing.id),
                "status": existing.status,
            },
        )


async def create_record(
    session: AsyncSession,
    *,
    x_user_id: str,
    handle: str | None = None,
    goal: str | None = None,
    tone: str | None = None,
    status: OutreachStatus = OutreachStatus.DRAFT,
    is_reply: bool = False,
) -> OutreachRecord:
    """Create a new outreach record, enforcing the dedup lock.

    The database has a partial unique index that guarantees at most one live
    record per recipient. This function translates the resulting integrity error
    into a clear domain error.
    """
    record = OutreachRecord(
        id=uuid.uuid4(),
        x_user_id=x_user_id,
        handle=handle,
        goal=goal,
        tone=tone,
        status=status.value,
        is_reply=is_reply,
    )
    session.add(record)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise DuplicateOutreachError(
            "An outreach record already exists for this account",
            details={"x_user_id": x_user_id},
        ) from exc
    return record


async def get_record(
    session: AsyncSession, record_id: uuid.UUID
) -> OutreachRecord | None:
    return await session.get(OutreachRecord, record_id)
