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
    parse_seed_handle,
)
from app.config import Settings, get_settings
from app.models import BotVerdict, DiscoveredAccount, DiscoveryRun
from app.services.cleanup import purge_discovery_history
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
#: Largest page requested from ``/2/users/search`` in one call.
USER_SEARCH_PAGE_MAX = 100
#: Largest page requested from the tweet-search endpoints in one call.
POST_SEARCH_PAGE_MAX = 100


def _page_size(remaining: int, page_max: int) -> int:
    """Return the page size that still fits in the remaining candidate budget.

    The X search endpoints always return up to ``max_results`` items, so a
    request larger than the remaining budget reads far more than the caller
    asked for. The floor is the smallest value X accepts for the endpoint,
    which the client raises internally when needed.
    """
    if remaining <= 0:
        return 0
    return min(remaining, page_max)


#: The reason recorded on an account the operator vouched for.
KNOWN_HUMAN_REASON = (
    "Marked as a known human by the operator; bot detection was skipped."
)


class KnownHumanMatcher:
    """Matches accounts the operator vouched for as human.

    The operator can name accounts they personally know are real people. Those
    accounts skip bot detection entirely, because a bot heuristic that fires
    on a genuine account would otherwise hide it from the results.

    The names are matched on the handle, so ``https://x.com/someone``,
    ``@someone`` and ``someone`` all refer to the same account. The match is
    case-insensitive, because X handles are.
    """

    def __init__(self, entries: list[str] | None = None) -> None:
        self._handles: set[str] = set()
        for entry in entries or []:
            handle = parse_seed_handle(entry)
            if handle:
                self._handles.add(handle.lower())

    def __bool__(self) -> bool:
        return bool(self._handles)

    @property
    def handles(self) -> set[str]:
        """The lowercased handles the operator vouched for."""
        return set(self._handles)

    def matches(self, *, username: str | None, x_user_id: str | None = None) -> bool:
        """Whether this account is one the operator vouched for."""
        if not self._handles:
            return False
        if username and username.strip().lower() in self._handles:
            return True
        # A numeric id is matched when the operator pasted one directly.
        return bool(x_user_id and x_user_id.strip() in self._handles)


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
    #: "seed" when the account was supplied by the caller, "search" when the
    #: X search found it. A seed that the search also returns keeps "search".
    origin: str = "search"


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
    #: How many of the supplied seed accounts made it into the results.
    seeds_included: int = 0
    #: How many supplied seed accounts resolved to a real X account.
    seeds_resolved: int = 0
    #: How many supplied seed accounts could not be resolved on X.
    seeds_unresolved: int = 0
    #: How many resolved seeds did not fit within max_accounts.
    seeds_dropped: int = 0


def _seed_to_candidate(profile: SeedProfile) -> Candidate | None:
    """Turn a resolved seed profile into a candidate.

    The seed's posts were already read while the segment was derived, so they
    are reused here instead of being fetched again. This keeps the seed
    inclusion free of extra X API cost.
    """
    if not profile.resolved or not profile.x_user_id:
        return None

    metrics = {
        "followers_count": profile.followers_count,
        "following_count": profile.following_count,
    }
    return Candidate(
        x_user_id=profile.x_user_id,
        profile={
            "id": profile.x_user_id,
            "username": profile.username,
            "name": profile.display_name,
            "description": profile.bio,
            "public_metrics": metrics,
        },
        posts=list(profile.posts),
        posts_read=len(profile.posts),
        users_read=1,
        origin="seed",
    )


def _estimate_cost(limits: SearchLimits, posts_read: int, users_read: int) -> float:
    return (
        posts_read * limits.post_read_cost_usd + users_read * limits.user_read_cost_usd
    )


async def run_discovery(
    session: AsyncSession,
    client: XClient,
    *,
    run_id: uuid.UUID | None = None,
    target_segment: list[str] | None = None,
    seed_accounts: list[str] | None = None,
    limits: SearchLimits | None = None,
    settings: Settings | None = None,
    include_bots: bool = False,
    bot_score_threshold: float = 0.6,
    known_human_accounts: list[str] | None = None,
) -> DiscoveryResult:
    """Run the full discovery flow and persist the results.

    ``run_id`` reuses a run row that was created before the work started, so
    the caller can poll that run while the search is still going. Without it
    a new run is created here.

    ``known_human_accounts`` names accounts the operator vouched for. A
    matching account skips bot detection and is always treated as human.
    """
    settings = settings or get_settings()
    limits = limits or SearchLimits(
        spend_ceiling_usd=settings.discovery_spend_ceiling_usd,
        post_read_cost_usd=settings.post_read_cost_usd,
        user_read_cost_usd=settings.user_read_cost_usd,
    )

    # Every new search starts from an empty history: the previous runs and
    # the accounts they produced are removed first.
    await purge_discovery_history(session)

    if run_id is None:
        run = DiscoveryRun(
            input_type=_input_type(target_segment, seed_accounts),
            target_segment=list(target_segment or []),
            seed_accounts=list(seed_accounts or []),
            status="RUNNING",
        )
        session.add(run)
    else:
        existing = await session.get(DiscoveryRun, run_id)
        if existing is not None:
            run = existing
        else:
            run = DiscoveryRun(
                id=run_id,
                input_type=_input_type(target_segment, seed_accounts),
                target_segment=list(target_segment or []),
                seed_accounts=list(seed_accounts or []),
                status="RUNNING",
            )
            session.add(run)
    await session.flush()

    warnings: list[str] = []
    seed_profiles: list[SeedProfile] = []
    bots_filtered = 0

    # The accounts the operator vouched for skip bot detection entirely.
    known_humans = KnownHumanMatcher(known_human_accounts)
    humans_bypassed = 0
    if known_humans:
        log.info(
            "known_human_accounts_supplied",
            count=len(known_humans.handles),
        )

    # ---- Stage 1: the seed accounts -----------------------------------
    # The seed accounts are candidates in their own right: the caller named
    # them explicitly, so they belong in the results even when no segment
    # could be derived from them.
    seed_profiles = await collect_seed_profiles(client, seed_accounts or [])
    seed_candidates = [
        candidate
        for candidate in (
            _seed_to_candidate(profile) for profile in seed_profiles
        )
        if candidate is not None
    ]

    # A suppressed account must never appear in discovery results, so the
    # seeds are checked against the opt-out list before they are counted.
    # A suppressed seed frees its slot for a searched account.
    suppressed_ids = await _suppressed_user_ids(session)
    seeds = [
        candidate
        for candidate in seed_candidates
        if candidate.x_user_id not in suppressed_ids
    ]

    # ---- Stage 2: bot detection for the seeds --------------------------
    # The seeds are assessed before the search, because their verdict decides
    # how many slots the search may still fill: a seed that is filtered out
    # must not consume one of the ``max_accounts`` slots.
    #
    # A seed the operator vouched for is never assessed: it is known to be
    # human, so the bot heuristics cannot override that.
    seed_assessments: dict[str, BotAssessment] = {}
    surviving_seeds: list[Candidate] = []
    for candidate in seeds:
        if known_humans.matches(
            username=str(candidate.profile.get("username") or ""),
            x_user_id=candidate.x_user_id,
        ):
            assessment = BotAssessment(
                score=0.0,
                verdict=BotVerdict.HUMAN.value,
                reasons=[KNOWN_HUMAN_REASON],
            )
            humans_bypassed += 1
            log.info(
                "known_human_bypassed_bot_detection",
                x_user_id=candidate.x_user_id,
                origin=candidate.origin,
            )
        else:
            assessment = await assess_account(
                candidate.profile,
                candidate.posts,
                threshold=bot_score_threshold,
            )
        seed_assessments[candidate.x_user_id] = assessment
        if assessment.verdict == BotVerdict.LIKELY_BOT.value and not include_bots:
            bots_filtered += 1
            continue
        surviving_seeds.append(candidate)

    # The seeds come first, so they are truncated to ``max_accounts`` in the
    # order the caller supplied them.
    dropped_seeds = max(0, len(surviving_seeds) - limits.max_accounts)
    if dropped_seeds > 0:
        surviving_seeds = surviving_seeds[: limits.max_accounts]
        warnings.append(
            f"{dropped_seeds} seed account(s) did not fit within the "
            f"max_accounts limit of {limits.max_accounts}."
        )

    # The remaining slots are what the X search may fill.
    remaining = max(0, limits.max_accounts - len(surviving_seeds))

    # ---- Stage 3: target segment ---------------------------------------
    segment: TargetSegment
    if seed_accounts:
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
        if not seed_candidates:
            run.status = "FAILED"
            run.stats = {"error": "Could not derive a target segment."}
            await session.flush()
            return DiscoveryResult(
                run_id=run.id,
                target_segment=segment,
                query_plan=QueryPlan(),
                warnings=warnings + ["No usable target segment could be derived."],
            )

        # Seeds alone are enough to answer the request: there is no segment
        # to search with, so the search stage is skipped entirely.
        warnings.append(
            "No usable target segment could be derived; returning the seed "
            "accounts only."
        )
        segment = TargetSegment(
            keywords=[],
            exclude_keywords=list(segment.exclude_keywords),
            rationale=segment.rationale,
            source=segment.source,
        )

    # ---- Stage 4: capabilities and the search --------------------------
    caps = Capabilities()
    query_plan = QueryPlan()

    # The seed accounts already fill every slot, so no X request is made at
    # all: the query plan, the capability probe and the search itself are
    # all skipped.
    candidates: list[Candidate] = []
    posts_read = 0
    users_read = 0
    api_calls = dict(client.call_counts)
    search_failures: list[str] = []

    if remaining == 0:
        log.info("discovery_search_skipped_seeds_fill_budget")
        warnings.append(
            "The seed accounts fill the max_accounts limit, so no X search "
            "was performed."
        )
        query_plan = QueryPlan(
            rationale="The seed accounts filled max_accounts; no search was needed."
        )
    else:
        # ---- Stage 4a: query plan ----------------------------------------
        query_plan = (
            await build_query_plan(segment.keywords, segment.exclude_keywords)
            if segment.keywords
            else QueryPlan(
                rationale="No segment was derived; the search stage was skipped."
            )
        )

        # ---- Stage 4b: capabilities -------------------------------------
        try:
            caps = await capability_probe.probe(client)
        except XApiError as exc:
            caps = Capabilities()
            log.warning("capability_probe_failed", **exc.log_fields())
            detail = exc.log_fields().get("x_error")
            suffix = f": {detail}" if detail else ""
            warnings.append(f"Capability probe failed{suffix}: {exc}")

        # The capability probe decides *which* tweet-search endpoint may be
        # used. When neither is available the post-search stage is skipped
        # entirely rather than firing requests at an endpoint X has already
        # refused.
        search_mode, mode_note = _resolve_search_mode(caps)
        if mode_note:
            warnings.append(mode_note)

        # ---- Stage 4c: the search ---------------------------------------
        if segment.keywords:
            (
                candidates,
                posts_read,
                users_read,
                api_calls,
                search_failures,
            ) = await _search_candidates(
                client,
                query_plan,
                limits,
                full_archive=search_mode == "full_archive",
                search_posts_enabled=search_mode != "none",
                max_results=remaining,
            )
            warnings.extend(search_failures)
        else:
            # Without a segment there is nothing to search for, so no X
            # request is made. The seed accounts are the whole answer.
            api_calls = dict(client.call_counts)

    cost = _estimate_cost(limits, posts_read, users_read)
    if cost > limits.spend_ceiling_usd:
        warnings.append(
            f"Estimated read cost {cost:.2f} USD exceeded the "
            f"{limits.spend_ceiling_usd:.2f} USD ceiling for this run."
        )
        candidates = candidates[: max(0, limits.max_candidates // 2)]

    # ---- Stage 7: bot detection and key information --------------------
    results: list[dict[str, Any]] = []

    # The seeds come first: the caller named them explicitly, so they are
    # the most relevant accounts the run can return. A seed the search also
    # found is not repeated.
    seed_ids = {candidate.x_user_id for candidate in surviving_seeds}
    searched = [
        candidate for candidate in candidates if candidate.x_user_id not in seed_ids
    ]
    ordered_candidates = [*surviving_seeds, *searched]

    for candidate in ordered_candidates:
        # A suppressed account must never appear in discovery results.
        if candidate.x_user_id in suppressed_ids:
            continue

        seed_assessment = seed_assessments.get(candidate.x_user_id)
        account_assessment: BotAssessment
        if seed_assessment is not None:
            # The seed was already assessed before the search, so the verdict
            # is reused instead of paying for the same LLM call twice.
            account_assessment = seed_assessment
        elif known_humans.matches(
            username=str(candidate.profile.get("username") or ""),
            x_user_id=candidate.x_user_id,
        ):
            # The operator vouched for this account, so it is not assessed.
            account_assessment = BotAssessment(
                score=0.0,
                verdict=BotVerdict.HUMAN.value,
                reasons=[KNOWN_HUMAN_REASON],
            )
            humans_bypassed += 1
            log.info(
                "known_human_bypassed_bot_detection",
                x_user_id=candidate.x_user_id,
                origin=candidate.origin,
            )
        else:
            account_assessment = await assess_account(
                candidate.profile,
                candidate.posts,
                threshold=bot_score_threshold,
            )
        is_bot = account_assessment.verdict == BotVerdict.LIKELY_BOT.value

        if is_bot and not include_bots:
            bots_filtered += 1
            await _persist_account(
                session,
                run.id,
                candidate,
                account_assessment,
                KeyInformation(),
                segment.keywords,
            )
            continue

        key_info = await extract_key_information(candidate.profile, candidate.posts)

        await _persist_account(
            session,
            run.id,
            candidate,
            account_assessment,
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
                "bot": account_assessment.to_payload(),
                "key_information": key_info.to_payload(),
                "origin": candidate.origin,
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
        "seeds_supplied": len(seed_accounts or []),
        "seeds_resolved": len(seed_candidates),
        "seeds_included": len(surviving_seeds),
        "seeds_dropped": dropped_seeds,
        "humans_bypassed": humans_bypassed,
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
        seeds_included=len(surviving_seeds),
        seeds_resolved=len(seed_candidates),
        seeds_unresolved=len(seed_accounts or []) - len(seed_candidates),
        seeds_dropped=dropped_seeds,
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
    max_results: int | None = None,
) -> tuple[list[Candidate], int, int, dict[str, int], list[str]]:
    """Run the query plan and hydrate the candidates it returns.

    Returns the candidates, the post and user read counters, the per-endpoint
    API call counts and a list of human-readable failure messages so the caller
    can surface partial failures instead of silently returning fewer results.

    ``max_results`` bounds how many candidates the search may contribute. The
    seed accounts already occupy the first slots of ``max_accounts``, so the
    search only has to look for the accounts that still fit. It never exceeds
    ``max_candidates``.
    """
    budget = limits.max_candidates if max_results is None else min(
        max_results, limits.max_candidates
    )
    candidate_ids: list[str] = []
    seen: set[str] = set()
    posts_read = 0
    users_read = 0
    #: Human-readable descriptions of every stage that produced no usable data.
    failures: list[str] = []

    # -- user search ------------------------------------------------------
    for query in query_plan.user_queries:
        # The page size is capped by the remaining candidate budget so a small
        # ``max_candidates`` does not still pull a full page of 100 accounts.
        page_size = _page_size(budget - len(candidate_ids), USER_SEARCH_PAGE_MAX)
        if page_size <= 0:
            break
        next_token: str | None = None
        for _ in range(MAX_PAGES_PER_QUERY):
            try:
                data = await client.search_users(
                    query, max_results=page_size, next_token=next_token
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
            if not next_token or len(candidate_ids) >= budget:
                break
        if len(candidate_ids) >= budget:
            break

    # -- tweet search -----------------------------------------------------
    # Skipped entirely when the capability probe found no usable endpoint.
    if search_posts_enabled:
        for query in query_plan.post_queries:
            # Tweet search pages cost a full ``max_results`` of reads, so the
            # page size is bounded by the remaining candidate budget.
            page_size = _page_size(budget - len(candidate_ids), POST_SEARCH_PAGE_MAX)
            if page_size <= 0:
                break
            next_token = None
            for _ in range(MAX_PAGES_PER_QUERY):
                try:
                    data = await client.search_posts(
                        query,
                        max_results=page_size,
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
                if not next_token or len(candidate_ids) >= budget:
                    break
            if len(candidate_ids) >= budget:
                break
    else:
        log.info("post_search_skipped_capability_unavailable")

    candidate_ids = candidate_ids[:budget]

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
    # ``max_posts_per_account=0`` means "read no posts": the call is skipped
    # entirely so no read cost is incurred at all.
    if limits.max_posts_per_account > 0:
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
                failures.append(
                    f"Reading posts for {candidate.x_user_id} failed: {exc}"
                )

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
    # A seed account is tagged so the caller can tell it apart from an
    # account the X search found.
    if candidate.origin == "seed" and "seed" not in account.segment_tags:
        account.segment_tags.append("seed")
    # An account the operator vouched for is tagged too, so it is visible in
    # the results that bot detection was skipped for it.
    if (
        assessment.verdict == BotVerdict.HUMAN.value
        and assessment.reasons
        and KNOWN_HUMAN_REASON in assessment.reasons
        and "known_human" not in account.segment_tags
    ):
        account.segment_tags.append("known_human")
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
