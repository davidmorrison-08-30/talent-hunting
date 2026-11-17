"""The orchestrated discovery search loop.

This module runs the search, hydrates candidates, collects evidence, applies bot
detection and extracts key information. It also enforces the pay-per-use spend
ceiling so a single run cannot run up an unexpected bill.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.botdetect import BotAssessment, assess_account
from app.agents.profile import KeyInformation, extract_key_information
from app.agents.queries import QueryPlan, build_query_plan
from app.agents.segment import (
    SeedProfile,
    TargetSegment,
    collect_seed_profiles,
    derive_target_segment,
)
from app.config import Settings, get_settings
from app.models import BotVerdict, DiscoveredAccount, DiscoveryRun
from app.x.capabilities import Capabilities, capability_probe
from app.x.client import XClient
from app.x.errors import XApiError

log = structlog.get_logger(__name__)

#: Maximum pages fetched per query.
MAX_PAGES_PER_QUERY = 2
#: Maximum candidates considered in one run.
DEFAULT_MAX_CANDIDATES = 150
#: Maximum accounts returned in one run.
DEFAULT_MAX_ACCOUNTS = 40


@dataclass
class SearchLimits:
    """Bounds applied to one discovery run."""

    max_accounts: int = DEFAULT_MAX_ACCOUNTS
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_posts_per_account: int = 20
    spend_ceiling_usd: float = 5.0
    post_read_cost_usd: float = 0.005
    user_read_cost_usd: float = 0.010


@dataclass
class Candidate:
    """A candidate account with its fetched evidence."""

    x_user_id: str
    profile: dict[str, Any]
    posts: list[dict[str, Any]] = field(default_factory=list)
    posts_read: int = 0
    users_read: int = 1
    source_queries: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """The outcome of a discovery run."""

    run_id: uuid.UUID
    target_segment: TargetSegment
    query_plan: QueryPlan
    accounts: list[dict[str, Any]] = field(default_factory=list)
    candidates_seen: int = 0
    bots_filtered: int = 0
    api_calls: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    capabilities: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _estimate_cost(limits: SearchLimits, posts_read: int, users_read: int) -> float:
    return (
        posts_read * limits.post_read_cost_usd + users_read * limits.user_read_cost_usd
    )


async def run_discovery(
    session: AsyncSession,
    client: XClient,
    *,
    target_segment: list[str] | None = None,
    seed_accounts: list[str] | None = None,
    limits: SearchLimits | None = None,
    settings: Settings | None = None,
    include_bots: bool = False,
    bot_score_threshold: float = 0.6,
) -> DiscoveryResult:
    """Run the full discovery flow and persist the results."""
    settings = settings or get_settings()
    limits = limits or SearchLimits(
        spend_ceiling_usd=settings.discovery_spend_ceiling_usd,
        post_read_cost_usd=settings.post_read_cost_usd,
        user_read_cost_usd=settings.user_read_cost_usd,
    )

    run = DiscoveryRun(
        input_type=_input_type(target_segment, seed_accounts),
        target_segment=list(target_segment or []),
        seed_accounts=list(seed_accounts or []),
        status="RUNNING",
    )
    session.add(run)
    await session.flush()

    warnings: list[str] = []
    seed_profiles: list[SeedProfile] = []

    # ---- Stage 1: target segment ---------------------------------------
    segment: TargetSegment
    if seed_accounts:
        seed_profiles = await collect_seed_profiles(client, seed_accounts)
        derived = await derive_target_segment(seed_profiles)
        segment = derived.merged_with(target_segment or [])
        for profile in seed_profiles:
            if not profile.resolved:
                warnings.append(f"Seed {profile.username}: {profile.error}")
    else:
        segment = TargetSegment(
            keywords=[k.strip() for k in (target_segment or []) if k.strip()],
            exclude_keywords=[],
            rationale="Segment supplied directly by the caller.",
            source="provided",
        )

    if not segment.keywords:
        run.status = "FAILED"
        run.stats = {"error": "Could not derive a target segment."}
        await session.flush()
        return DiscoveryResult(
            run_id=run.id,
            target_segment=segment,
            query_plan=QueryPlan(),
            warnings=warnings + ["No usable target segment could be derived."],
        )

    # ---- Stage 2: query plan -------------------------------------------
    query_plan = await build_query_plan(segment.keywords, segment.exclude_keywords)

    # ---- Capabilities ---------------------------------------------------
    try:
        caps: Capabilities = await capability_probe.probe(client)
    except XApiError as exc:
        caps = Capabilities()
        log.warning("capability_probe_failed", **exc.log_fields())
        detail = exc.log_fields().get("x_error")
        suffix = f": {detail}" if detail else ""
        warnings.append(f"Capability probe failed{suffix}: {exc}")

    # ---- Stage 3: search -----------------------------------------------
    # The capability probe decides *which* tweet-search endpoint may be used.
    # When neither is available the post-search stage is skipped entirely
    # rather than firing requests at an endpoint X has already refused.
    search_mode, mode_note = _resolve_search_mode(caps)
    if mode_note:
        warnings.append(mode_note)

    candidates, posts_read, users_read, api_calls, search_failures = (
        await _search_candidates(
            client,
            query_plan,
            limits,
            full_archive=search_mode == "full_archive",
            search_posts_enabled=search_mode != "none",
        )
    )
    warnings.extend(search_failures)

    cost = _estimate_cost(limits, posts_read, users_read)
    if cost > limits.spend_ceiling_usd:
        warnings.append(
            f"Estimated read cost {cost:.2f} USD exceeded the "
            f"{limits.spend_ceiling_usd:.2f} USD ceiling for this run."
        )
        candidates = candidates[: max(0, limits.max_candidates // 2)]

    # ---- Stage 4-5: bot detection and key information ------------------
    results: list[dict[str, Any]] = []
    bots_filtered = 0
    suppressed_ids = await _suppressed_user_ids(session)

    for candidate in candidates:
        # A suppressed account must never appear in discovery results.
        if candidate.x_user_id in suppressed_ids:
            continue

        assessment = await assess_account(
            candidate.profile,
            candidate.posts,
            threshold=bot_score_threshold,
        )
        is_bot = assessment.verdict == BotVerdict.LIKELY_BOT.value

        if is_bot and not include_bots:
            bots_filtered += 1
            await _persist_account(
                session,
                run.id,
                candidate,
                assessment,
                KeyInformation(),
                segment.keywords,
            )
            continue

        key_info = await extract_key_information(candidate.profile, candidate.posts)

        await _persist_account(
            session,
            run.id,
            candidate,
            assessment,
            key_info,
            segment.keywords,
        )

        results.append(
            {
                "x_user_id": candidate.x_user_id,
                "username": candidate.profile.get("username"),
                "display_name": candidate.profile.get("name"),
                "profile_url": f"https://x.com/{candidate.profile.get('username')}",
                "bio": candidate.profile.get("description") or "",
                "followers_count": int(
                    (candidate.profile.get("public_metrics") or {}).get(
                        "followers_count", 0
                    )
                ),
                "following_count": int(
                    (candidate.profile.get("public_metrics") or {}).get(
                        "following_count", 0
                    )
                ),
                "created_at": candidate.profile.get("created_at"),
                "bot": assessment.to_payload(),
                "key_information": key_info.to_payload(),
            }
        )

        if len(results) >= limits.max_accounts:
            break

    run.status = "COMPLETED"
    run.target_segment = segment.keywords
    run.stats = {
        "candidates": len(candidates),
        "bots_filtered": bots_filtered,
        "returned": len(results),
        "api_calls": api_calls,
        "posts_read": posts_read,
        "users_read": users_read,
        "estimated_cost_usd": round(cost, 4),
        "warnings": warnings,
    }
    await session.flush()

    return DiscoveryResult(
        run_id=run.id,
        target_segment=segment,
        query_plan=query_plan,
        accounts=results,
        candidates_seen=len(candidates),
        bots_filtered=bots_filtered,
        api_calls=api_calls,
        estimated_cost_usd=cost,
        capabilities=caps.to_payload(),
        warnings=warnings,
    )


def _input_type(target_segment: list[str] | None, seed_accounts: list[str] | None) -> str:
    if target_segment and seed_accounts:
        return "both"
    if seed_accounts:
        return "seed_accounts"
    return "target_segment"


#: The tweet-search endpoint a discovery run should use.
#:
#: ``"full_archive"`` -> ``/2/tweets/search/all``
#: ``"recent"``      -> ``/2/tweets/search/recent``
#: ``"none"``        -> skip tweet search entirely
SearchMode = Literal["full_archive", "recent", "none"]


def _resolve_search_mode(caps: Capabilities) -> tuple[SearchMode, str]:
    """Pick the tweet-search endpoint from the probed capabilities.

    The probe already knows which endpoints X will accept. Re-probing a
    refused endpoint during the search stage wastes budget and produces a
    stream of identical 400/403 warnings, so the search stage follows the
    probe's verdict instead of retrying blindly.
    """
    if caps.full_archive_search:
        return "full_archive", ""
    if caps.recent_search:
        return "recent", ""
    return (
        "none",
        "Tweet search is unavailable on this X account "
        "(both /2/tweets/search/recent and /2/tweets/search/all were refused). "
        "Discovery is limited to user search results.",
    )


async def _search_candidates(
    client: XClient,
    query_plan: QueryPlan,
    limits: SearchLimits,
    *,
    full_archive: bool,
    search_posts_enabled: bool = True,
) -> tuple[list[Candidate], int, int, dict[str, int], list[str]]:
    """Run the query plan and hydrate the candidates it returns.

    Returns the candidates, the post and user read counters, the per-endpoint
    API call counts and a list of human-readable failure messages so the caller
    can surface partial failures instead of silently returning fewer results.
    """
    candidate_ids: list[str] = []
    seen: set[str] = set()
    posts_read = 0
    users_read = 0
    #: Human-readable descriptions of every stage that produced no usable data.
    failures: list[str] = []

    # -- user search ------------------------------------------------------
    for query in query_plan.user_queries:
        next_token: str | None = None
        for _ in range(MAX_PAGES_PER_QUERY):
            try:
                data = await client.search_users(
                    query, max_results=100, next_token=next_token
                )
            except XApiError as exc:
                log.warning("user_search_failed", query=query, **exc.log_fields())
                failures.append(f"User search failed for {query!r}: {exc}")
                break

            for user in data.get("data") or []:
                uid = str(user.get("id") or "")
                if uid and uid not in seen:
                    seen.add(uid)
                    candidate_ids.append(uid)

            next_token = (data.get("meta") or {}).get("next_token")
            if not next_token or len(candidate_ids) >= limits.max_candidates:
                break

    # -- tweet search -----------------------------------------------------
    # Skipped entirely when the capability probe found no usable endpoint.
    if search_posts_enabled:
        for query in query_plan.post_queries:
            next_token = None
            for _ in range(MAX_PAGES_PER_QUERY):
                try:
                    data = await client.search_posts(
                        query,
                        max_results=100,
                        next_token=next_token,
                        full_archive=full_archive,
                    )
                except XApiError as exc:
                    log.warning("post_search_failed", query=query, **exc.log_fields())
                    failures.append(f"Tweet search failed for {query!r}: {exc}")
                    break

                for post in data.get("data") or []:
                    author = post.get("author_id")
                    if author and str(author) not in seen:
                        seen.add(str(author))
                        candidate_ids.append(str(author))
                    posts_read += 1

                next_token = (data.get("meta") or {}).get("next_token")
                if not next_token or len(candidate_ids) >= limits.max_candidates:
                    break
    else:
        log.info("post_search_skipped_capability_unavailable")

    candidate_ids = candidate_ids[: limits.max_candidates]

    # -- hydrate ----------------------------------------------------------
    candidates: list[Candidate] = []
    for index in range(0, len(candidate_ids), 100):
        batch = candidate_ids[index : index + 100]
        try:
            profiles = await client.get_users_by_ids(batch)
        except XApiError as exc:
            log.warning("hydrate_failed", **exc.log_fields())
            failures.append(f"Hydrating {len(batch)} candidate(s) failed: {exc}")
            continue
        users_read += len(profiles)

        for profile in profiles:
            candidates.append(
                Candidate(x_user_id=str(profile.get("id") or ""), profile=profile)
            )

    # -- fetch evidence ----------------------------------------------------
    for candidate in candidates:
        try:
            tweets = await client.get_user_tweets(
                candidate.x_user_id,
                max_results=limits.max_posts_per_account,
            )
            candidate.posts = list(tweets.get("data") or [])
            candidate.posts_read = len(candidate.posts)
            posts_read += candidate.posts_read
        except XApiError as exc:
            log.warning(
                "candidate_posts_failed",
                x_user_id=candidate.x_user_id,
                **exc.log_fields(),
            )
            failures.append(f"Reading posts for {candidate.x_user_id} failed: {exc}")

    api_calls = dict(client.call_counts)
    return candidates, posts_read, users_read, api_calls, failures


async def _suppressed_user_ids(session: AsyncSession) -> set[str]:
    """The user ids on the permanent opt-out list."""
    from app.models import SuppressionEntry

    result = await session.execute(select(SuppressionEntry.x_user_id))
    return {str(uid) for uid in result.scalars().all() if uid}


async def _persist_account(
    session: AsyncSession,
    run_id: uuid.UUID,
    candidate: Candidate,
    assessment: BotAssessment,
    key_info: KeyInformation,
    segment_tags: list[str],
) -> None:
    """Upsert a discovered account with its evidence and key information."""
    profile = candidate.profile
    metrics = profile.get("public_metrics") or {}

    result = await session.execute(
        select(DiscoveredAccount).where(
            DiscoveredAccount.x_user_id == candidate.x_user_id
        )
    )
    account = result.scalar_one_or_none()

    if account is None:
        account = DiscoveredAccount(x_user_id=candidate.x_user_id, username="")
        session.add(account)

    account.username = str(profile.get("username") or "")
    account.display_name = profile.get("name")
    account.bio = profile.get("description") or ""
    account.followers_count = int(metrics.get("followers_count") or 0)
    account.following_count = int(metrics.get("following_count") or 0)
    account.tweets_count = int(metrics.get("tweet_count") or 0)
    account.created_at_x = _parse_datetime(profile.get("created_at"))
    account.verified = bool(profile.get("verified"))
    account.avatar_url = profile.get("profile_image_url")
    account.public_metrics = metrics
    account.bot_score = assessment.score
    account.bot_verdict = assessment.verdict
    account.bot_reasons = assessment.reasons
    account.key_info = key_info.to_payload()
    account.segment_tags = list(segment_tags)
    account.source_run_id = run_id
    account.last_seen_at = datetime.now(UTC)

    await session.flush()


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
