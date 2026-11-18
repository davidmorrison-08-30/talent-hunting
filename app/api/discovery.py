"""PRIMARY ENDPOINT I: discovery.

``POST /api/v1/discovery/search`` finds accounts on X that match a target
segment and/or seed accounts, filters out bots, and returns each surviving
account with its key information.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.search import SearchLimits
from app.config import Settings, get_settings
from app.db import get_session
from app.errors import NotFoundError, ValidationError
from app.models import DiscoveredAccount, DiscoveryRun, SuppressionEntry
from app.schemas import (
    AccountRecord,
    DiscoveryAccepted,
    DiscoveryRequest,
    DiscoveryRunSummary,
)
from app.schemas import (
    KeyInformation as KeyInformationSchema,
)
from app.services.search_runner import start_run

router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])


def _to_account_record(account: DiscoveredAccount) -> AccountRecord:
    """Map a stored account row onto the public response model."""
    return AccountRecord(
        x_user_id=account.x_user_id,
        username=account.username,
        display_name=account.display_name,
        bio=account.bio or "",
        followers_count=account.followers_count,
        following_count=account.following_count,
        tweets_count=account.tweets_count,
        verified=account.verified,
        created_at=account.created_at_x,
        bot_verdict=account.bot_verdict,
        bot_score=account.bot_score,
        bot_reasons=account.bot_reasons or [],
        key_information=KeyInformationSchema(**(account.key_info or {})),
        segment_tags=account.segment_tags or [],
        source_run_id=account.source_run_id,
        last_seen_at=account.last_seen_at,
    )


@router.post(
    "/search",
    response_model=DiscoveryAccepted,
    status_code=202,
    summary="Start a discovery search for accounts matching a segment and/or seeds",
)
async def search_accounts(
    request: DiscoveryRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> DiscoveryAccepted:
    """Start a discovery search and return the run to poll.

    The search waits on the X API and on several LLM inferences, which takes
    much longer than an API Gateway integration may hold a connection open.
    The request therefore only records the run and returns immediately; the
    caller polls ``GET /api/v1/discovery/runs/{run_id}`` until the run leaves
    the running state and then reads the accounts it found.
    """
    if not request.target_segment and not request.seed_accounts:
        raise ValidationError(
            "Provide at least one of 'target_segment' or 'seed_accounts'"
        )

    limits = SearchLimits(
        max_accounts=request.max_accounts,
        max_candidates=request.max_candidates,
        max_posts_per_account=request.max_posts_per_account,
        spend_ceiling_usd=settings.discovery_spend_ceiling_usd,
        post_read_cost_usd=settings.post_read_cost_usd,
        user_read_cost_usd=settings.user_read_cost_usd,
    )

    run_id = await start_run(
        session,
        target_segment=request.target_segment,
        seed_accounts=request.seed_accounts,
        limits=limits,
        settings=settings,
        include_bots=request.include_bots,
        bot_score_threshold=request.bot_score_threshold,
    )

    return DiscoveryAccepted(
        run_id=run_id,
        status="RUNNING",
        poll_url=f"/api/v1/discovery/runs/{run_id}",
    )


@router.get(
    "/runs/{run_id}",
    response_model=DiscoveryRunSummary,
    summary="Get a discovery run and its statistics",
)
async def get_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DiscoveryRunSummary:
    result = await session.execute(
        select(DiscoveryRun).where(DiscoveryRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise NotFoundError(f"No discovery run with id {run_id}")
    return DiscoveryRunSummary(
        id=run.id,
        input_type=run.input_type,
        target_segment=run.target_segment or [],
        seed_accounts=run.seed_accounts or [],
        status=run.status,
        stats=run.stats or {},
        created_at=run.created_at,
    )


@router.get(
    "/runs",
    response_model=list[DiscoveryRunSummary],
    summary="List recent discovery runs",
)
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> list[DiscoveryRunSummary]:
    result = await session.execute(
        select(DiscoveryRun).order_by(DiscoveryRun.created_at.desc()).limit(limit)
    )
    return [
        DiscoveryRunSummary(
            id=run.id,
            input_type=run.input_type,
            target_segment=run.target_segment or [],
            seed_accounts=run.seed_accounts or [],
            status=run.status,
            stats=run.stats or {},
            created_at=run.created_at,
        )
        for run in result.scalars().all()
    ]


@router.get(
    "/accounts",
    response_model=list[AccountRecord],
    summary="List accounts found by previous discovery runs",
)
async def list_accounts(
    bot_verdict: str | None = Query(
        default=None, description="Filter by verdict: human, likely_bot or unknown."
    ),
    min_followers: int = Query(default=0, ge=0),
    segment_tag: str | None = Query(default=None),
    include_suppressed: bool = Query(
        default=False,
        description="Include accounts that opted out and are on the suppression list.",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[AccountRecord]:
    query = select(DiscoveredAccount).where(
        DiscoveredAccount.followers_count >= min_followers
    )
    if bot_verdict:
        query = query.where(DiscoveredAccount.bot_verdict == bot_verdict)
    if segment_tag:
        query = query.where(DiscoveredAccount.segment_tags.contains([segment_tag]))
    if not include_suppressed:
        # A suppressed account is removed from the discovery results.
        suppressed = select(SuppressionEntry.x_user_id)
        query = query.where(DiscoveredAccount.x_user_id.not_in(suppressed))

    result = await session.execute(
        query.order_by(DiscoveredAccount.followers_count.desc()).limit(limit).offset(offset)
    )

    return [_to_account_record(account) for account in result.scalars().all()]


@router.get(
    "/accounts/{x_user_id}",
    response_model=AccountRecord,
    summary="Get one discovered account",
)
async def get_account(
    x_user_id: str,
    session: AsyncSession = Depends(get_session),
) -> AccountRecord:
    account = await session.get(DiscoveredAccount, x_user_id)
    if account is None:
        raise NotFoundError(f"No discovered account with id {x_user_id}")
    return _to_account_record(account)
