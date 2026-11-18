"""Data cleanup for disconnecting and for starting a fresh discovery run.

Two levels of cleanup are provided:

* :func:`clear_operator_data` wipes everything that belongs to the connected
  operator, so a disconnect leaves no trace of the account behind. The
  permanent opt-out list and the daily rate budget deliberately survive:
  the opt-outs protect other people from being contacted again, and the
  budget stops a reconnect from resetting the daily send and read caps.
* :func:`purge_discovery_history` removes the previous discovery runs and the
  accounts they produced, so a new search starts from an empty history.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    DiscoveredAccount,
    DiscoveryRun,
    DmConversation,
    DmEvent,
    OAuthState,
    OAuthToken,
    OutreachRecord,
)

log = structlog.get_logger(__name__)


def _row_count(result: Any) -> int:
    """Return the number of rows a DELETE affected."""
    rowcount = getattr(result, "rowcount", None)
    return int(rowcount or 0)


async def clear_operator_data(session: AsyncSession) -> dict[str, int]:
    """Delete every trace of the connected operator.

    The suppression list and the rate budget are intentionally left alone.
    They are not operator data: the opt-out list records people who asked not
    to be contacted, and the budget enforces the daily ceilings.
    """
    counts: dict[str, int] = {}

    # The DM event log and the conversation cache are keyed by recipient and
    # have no foreign keys, so they go first.
    for name, model in (
        ("dm_events", DmEvent),
        ("dm_conversations", DmConversation),
        ("outreach_records", OutreachRecord),
        ("discovered_accounts", DiscoveredAccount),
        ("discovery_runs", DiscoveryRun),
        ("oauth_tokens", OAuthToken),
        ("oauth_states", OAuthState),
    ):
        result = await session.execute(delete(model))
        counts[name] = _row_count(result)

    await session.flush()
    log.info("operator_data_cleared", **counts)
    return counts


async def purge_discovery_history(session: AsyncSession) -> dict[str, int]:
    """Delete the previous discovery runs and the accounts they produced.

    The accounts are deleted after the runs because
    ``discovered_accounts.source_run_id`` points at a run with
    ``ondelete="SET NULL"``: removing a run orphans its accounts rather than
    removing them, so a second pass clears the orphans.
    """
    run_result = await session.execute(delete(DiscoveryRun))

    # An account whose run is gone, or that was never linked to a run, is no
    # longer reachable from the run history and is removed with it.
    orphan_result = await session.execute(
        delete(DiscoveredAccount).where(
            or_(
                DiscoveredAccount.source_run_id.is_(None),
                ~DiscoveredAccount.source_run_id.in_(select(DiscoveryRun.id)),
            )
        )
    )

    await session.flush()
    counts = {
        "discovery_runs": _row_count(run_result),
        "discovered_accounts": _row_count(orphan_result),
    }
    log.info("discovery_history_purged", **counts)
    return counts
