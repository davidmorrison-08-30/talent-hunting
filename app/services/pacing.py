"""Dispatch pacing and the application-level daily DM ceiling.

``build-docs/x-rules-and-limits.md`` requires that approved messages are sent
sequentially with randomized human-like delays (30-90 seconds apart), and that
the application enforces a conservative daily cap well below X's own 500/day
account ceiling.

The daily count is persisted in Postgres so a restart does not reset the cap.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.errors import DailyCapError
from app.models import RateBudget

log = structlog.get_logger(__name__)


def today_key() -> str:
    """The UTC day key used to bucket the daily counters."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def get_budget(session: AsyncSession, day: str | None = None) -> RateBudget:
    """Fetch (or create) the budget row for a day."""
    key = day or today_key()
    result = await session.execute(select(RateBudget).where(RateBudget.day == key))
    budget = result.scalar_one_or_none()
    if budget is None:
        budget = RateBudget(day=key)
        session.add(budget)
        await session.flush()
    return budget


async def dm_sent_today(session: AsyncSession) -> int:
    """How many DMs have been sent so far today."""
    budget = await get_budget(session)
    return budget.dm_sent_count


async def assert_daily_cap_available(session: AsyncSession) -> None:
    """Raise when the application-level daily DM ceiling has been reached."""
    settings = get_settings()
    budget = await get_budget(session)
    if budget.dm_sent_count >= settings.dm_daily_cap:
        raise DailyCapError(
            "The daily direct message cap has been reached",
            details={
                "sent_today": budget.dm_sent_count,
                "daily_cap": settings.dm_daily_cap,
                "day": budget.day,
            },
        )


async def record_dm_sent(session: AsyncSession) -> int:
    """Count one sent DM against today's budget and return the new total."""
    budget = await get_budget(session)
    budget.dm_sent_count += 1
    await session.flush()
    log.info("dm_sent_counted", total=budget.dm_sent_count, day=budget.day)
    return budget.dm_sent_count


def next_delay_seconds(settings: Settings | None = None) -> float:
    """A randomized human-like delay between consecutive sends."""
    settings = settings or get_settings()
    low = max(0.0, settings.dm_min_delay_seconds)
    high = max(low, settings.dm_max_delay_seconds)
    return random.uniform(low, high)


async def wait_before_send(settings: Settings | None = None) -> float:
    """Sleep for the randomized pacing delay and return how long we waited."""
    settings = settings or get_settings()
    delay = next_delay_seconds(settings)
    if delay > 0:
        log.info("pacing_delay", seconds=round(delay, 1))
        await asyncio.sleep(delay)
    return delay


async def track_reads(
    session: AsyncSession, *, posts: int = 0, users: int = 0
) -> float:
    """Track pay-per-use read volume and return the running cost estimate."""
    settings = get_settings()
    budget = await get_budget(session)
    budget.posts_read_count += posts
    budget.users_read_count += users
    budget.estimated_cost_usd = (
        budget.posts_read_count * settings.post_read_cost_usd
        + budget.users_read_count * settings.user_read_cost_usd
    )
    await session.flush()
    return budget.estimated_cost_usd


async def track_dm_events_polled(session: AsyncSession) -> int:
    """Count one DM event poll against today's budget."""
    budget = await get_budget(session)
    budget.dm_events_polled += 1
    await session.flush()
    return budget.dm_events_polled
