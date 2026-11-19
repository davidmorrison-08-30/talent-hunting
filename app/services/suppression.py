"""The permanent opt-out suppression list.

A one-click "Do Not Contact" flag must block the account from future discovery
results as well as from any new outreach, and it must never be removed
automatically.
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DiscoveredAccount, OutreachRecord, OutreachStatus, SuppressionEntry

log = structlog.get_logger(__name__)


async def suppress(
    session: AsyncSession,
    *,
    x_user_id: str | None = None,
    handle: str | None = None,
    reason: str = "",
) -> SuppressionEntry:
    """Add an account to the suppression list.

    Any live outreach record for the account is moved to ``OPTED_OUT`` so the
    dedup lock keeps working and the history stays auditable.
    """
    if not x_user_id and not handle:
        raise ValueError("Either x_user_id or handle must be provided")

    resolved_id = x_user_id
    if not resolved_id and handle:
        resolved_id = await _resolve_handle(session, handle)

    if not resolved_id:
        raise ValueError(f"Could not resolve a user id for handle {handle!r}")

    existing = await session.get(SuppressionEntry, resolved_id)
    if existing is not None:
        existing.reason = reason or existing.reason
        existing.handle = handle or existing.handle
        await session.flush()
        log.info("suppression_reaffirmed", x_user_id=resolved_id)
        return existing

    entry = SuppressionEntry(
        x_user_id=resolved_id,
        handle=handle,
        reason=reason or None,
    )
    session.add(entry)

    # Move any live outreach record to OPTED_OUT.
    result = await session.execute(
        select(OutreachRecord).where(
            OutreachRecord.x_user_id == resolved_id,
            OutreachRecord.status.in_(
                (
                    OutreachStatus.DRAFT.value,
                    OutreachStatus.PENDING_APPROVAL.value,
                )
            ),
        )
    )
    for record in result.scalars().all():
        record.status = OutreachStatus.OPTED_OUT.value

    await session.flush()
    log.info("suppression_added", x_user_id=resolved_id, handle=handle)
    return entry


async def list_suppressions(
    session: AsyncSession, *, limit: int = 100, offset: int = 0
) -> list[SuppressionEntry]:
    result = await session.execute(
        select(SuppressionEntry)
        .order_by(SuppressionEntry.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def _resolve_handle(session: AsyncSession, handle: str) -> str | None:
    """Look up a user id by handle using the discovered-accounts table."""
    cleaned = handle.strip().lstrip("@")
    result = await session.execute(
        select(DiscoveredAccount).where(DiscoveredAccount.username == cleaned).limit(1)
    )
    account = result.scalar_one_or_none()
    if account is not None:
        return account.x_user_id
    return None
