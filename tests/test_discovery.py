"""Tests for the discovery pipeline: segment, query plan, bot detection, search."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from app.agents.botdetect import assess_account, run_heuristics
from app.agents.queries import (
    build_query_plan,
    sanitize_post_query,
    sanitize_user_query,
)
from app.agents.search import SearchLimits, run_discovery
from app.agents.segment import parse_seed_handle
from app.x.capabilities import capability_probe
from app.x.client import XClient


class FakeLLM:
    """A BedrockLLM stand-in that returns canned structured responses.

    Responses can be supplied as an ordered list or as a mapping keyed by a
    substring of the prompt, which keeps tests independent of the exact order
    in which the agents call the model.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        by_prompt: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.by_prompt = dict(by_prompt or {})
        self.calls: list[dict[str, Any]] = []

    async def converse_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        user_prompt = str(kwargs.get("user", ""))
        system_prompt = str(kwargs.get("system", ""))
        for marker, response in self.by_prompt.items():
            if marker in user_prompt or marker in system_prompt:
                return response
        if self.responses:
            return self.responses.pop(0)
        return {}


def _profile(
    user_id: str = "1",
    username: str = "alice_robotics",
    **overrides: Any,
) -> dict[str, Any]:
    profile = {
        "id": user_id,
        "username": username,
        "name": "Alice Ng",
        "description": "Robotics engineer building legged robots. Previously research lab.",
        "created_at": "2013-05-02T00:00:00.000Z",
        "verified": True,
        "profile_image_url": "https://pbs.twimg.com/profile_images/1.jpg",
        "public_metrics": {
            "followers_count": 4200,
            "following_count": 800,
            "tweet_count": 5600,
        },
    }
    profile.update(overrides)
    return profile


def _profiles(count: int = 1) -> list[dict[str, Any]]:
    """Build ``count`` distinct profiles for the search endpoints."""
    return [_profile(user_id=str(i), username=f"user_{i}") for i in range(count)]


def _posts(count: int = 12) -> list[dict[str, Any]]:
    texts = [
        "Shipped a new IK solver today",
        "Anyone tried the new sensor suite?",
        "Reading about soft actuators, fascinating",
        "Conference talk accepted",
        "Finally got the arm to walk",
        "Reading a paper on diffusion policies",
        "Demo day was a blast",
        "Trying out a new simulator",
        "Anyone hiring for controls roles",
        "Conference prep",
        "New week new project",
        "Debugging all day",
    ]
    return [
        {
            "id": f"post-{i}",
            "created_at": (datetime.now(UTC) - timedelta(hours=i * 13)).isoformat(),
            "text": texts[i % len(texts)],
            "public_metrics": {
                "like_count": 12,
                "reply_count": 2,
                "retweet_count": 1,
                "quote_count": 0,
            },
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Seed handle parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://x.com/alice", "alice"),
        ("http://x.com/alice", "alice"),
        ("https://twitter.com/alice", "alice"),
        ("https://www.x.com/alice", "alice"),
        ("x.com/alice", "alice"),
        ("@alice", "alice"),
        ("alice", "alice"),
        ("alice_robotics", "alice_robotics"),
        ("https://x.com/alice?ref=1", "alice"),
    ],
)
def test_parse_seed_handle(raw: str, expected: str) -> None:
    assert parse_seed_handle(raw) == expected


@pytest.mark.parametrize("raw", ["", "https://example.com/alice", "not a url"])
def test_parse_seed_handle_rejects_bad_input(raw: str) -> None:
    assert parse_seed_handle(raw) is None


# ---------------------------------------------------------------------------
# Query sanitization
# ---------------------------------------------------------------------------
def test_sanitize_user_query_strips_operators() -> None:
    assert sanitize_user_query('robotics "engineer" -is:retweet') == "robotics engineer"


def test_sanitize_user_query_enforces_50_char_limit() -> None:
    cleaned = sanitize_user_query("a" * 80)
    assert cleaned is not None
    assert len(cleaned) == 50


def test_sanitize_user_query_returns_none_for_symbols_only() -> None:
    assert sanitize_user_query("!!! ??? ---") is None


def test_sanitize_post_query_rewrites_min_faves_to_min_likes() -> None:
    """X web-search syntax is rejected by the API with HTTP 400."""
    cleaned = sanitize_post_query('(AI OR LLM) min_faves:10 -is:retweet')

    assert cleaned is not None
    assert "min_faves:" not in cleaned
    assert "min_likes:10" in cleaned


def test_sanitize_post_query_rewrites_negated_min_faves() -> None:
    cleaned = sanitize_post_query("AI -min_faves:10")

    assert cleaned is not None
    assert "min_faves:" not in cleaned
    assert "-min_likes:10" in cleaned


def test_sanitize_post_query_rewrites_min_retweets() -> None:
    cleaned = sanitize_post_query("AI min_retweets:5")

    assert cleaned is not None
    assert "min_retweets:" not in cleaned
    assert "min_reposts:5" in cleaned


def test_sanitize_post_query_drops_a_query_without_a_positive_clause() -> None:
    """X rejects a negation-only query with ``Rules must contain a non-negation term``."""
    assert sanitize_post_query("-is:retweet") is None
    assert sanitize_post_query("-is:retweet -is:reply") is None


def test_sanitize_post_query_keeps_a_query_with_a_positive_clause() -> None:
    cleaned = sanitize_post_query("AI -is:retweet")

    assert cleaned is not None
    assert cleaned == "AI -is:retweet"


def test_sanitize_post_query_replaces_the_bare_and_keyword() -> None:
    """The API rejects a bare ``AND``; a space is the API's logical AND.

    X answers ``Ambiguous use of and as a keyword. Use a space to logically join
    two clauses, or "and" to find occurrences of and in text``.
    """
    cleaned = sanitize_post_query('"AI researcher" AND blog -is:retweet')

    assert cleaned is not None
    assert " AND " not in cleaned
    assert cleaned == '"AI researcher" blog -is:retweet'


def test_sanitize_post_query_replaces_the_bare_not_keyword() -> None:
    """The API rejects a bare ``NOT``; a leading hyphen is the negation."""
    cleaned = sanitize_post_query("AI NOT crypto")

    assert cleaned is not None
    assert "NOT" not in cleaned
    assert cleaned == "AI -crypto"


def test_sanitize_post_query_keeps_and_inside_a_quoted_phrase() -> None:
    """A quoted phrase is matched literally, so its words must not be rewritten."""
    cleaned = sanitize_post_query('"cats and dogs" -is:retweet')

    assert cleaned is not None
    assert '"cats and dogs"' in cleaned


def test_sanitize_post_query_does_not_rewrite_a_word_containing_and() -> None:
    """``ANDY`` is a term, not the ``AND`` operator."""
    cleaned = sanitize_post_query("ANDY -is:retweet")

    assert cleaned is not None
    assert "ANDY" in cleaned


def test_sanitize_post_query_keeps_the_uppercase_or_operator() -> None:
    """``OR`` is a valid API operator and must survive untouched."""
    cleaned = sanitize_post_query("(AI OR LLM) -is:retweet")

    assert cleaned is not None
    assert "OR" in cleaned


def test_sanitize_post_query_collapses_the_gap_left_by_and() -> None:
    """Removing ``AND`` must not leave a double space in the query."""
    cleaned = sanitize_post_query('"AI" AND "blog"')

    assert cleaned is not None
    assert "  " not in cleaned


@pytest.mark.asyncio
async def test_build_query_plan_sanitizes_llm_output() -> None:
    llm = FakeLLM(
        [
            {
                "user_queries": ["robotics engineer", "x" * 80, "!!!", "embodied ai"],
                "post_queries": ['("embodied ai" OR robotics) -is:retweet', "y" * 700],
                "rationale": "covers the segment",
            }
        ]
    )
    plan = await build_query_plan(["robotics"], llm=llm)

    assert "robotics engineer" in plan.user_queries
    assert "embodied ai" in plan.user_queries
    assert "!!!" not in plan.user_queries
    assert all(len(q) <= 50 for q in plan.user_queries)
    assert all(len(q) <= 512 for q in plan.post_queries)


@pytest.mark.asyncio
async def test_build_query_plan_falls_back_without_llm() -> None:
    class BrokenLLM:
        async def converse_json(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("bedrock down")

    plan = await build_query_plan(["robotics engineer"], llm=BrokenLLM())
    assert plan.user_queries
    assert plan.post_queries
    assert "-is:retweet" in plan.post_queries[0]


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------
def test_heuristics_flag_classic_bot() -> None:
    bot = _profile(
        username="cryptobot2024",
        description="crypto airdrop! click link in bio",
        created_at="2024-01-15T00:00:00.000Z",
        public_metrics={
            "followers_count": 10,
            "following_count": 9000,
            "tweet_count": 800,
        },
    )
    signals = run_heuristics(bot, [])
    names = {s.name for s in signals}
    assert "lopsided_follow_ratio" in names
    assert "single_issue_bio" in names
    assert "recent_account_high_volume" in names


def test_heuristics_do_not_flag_clean_human() -> None:
    signals = run_heuristics(_profile(), _posts())
    assert signals == []


def test_heuristics_flag_generated_handle_with_empty_bio() -> None:
    profile = _profile(
        username="John4821",
        description="",
        profile_image_url="",
        public_metrics={"followers_count": 5, "following_count": 20, "tweet_count": 3},
    )
    names = {s.name for s in run_heuristics(profile, [])}
    assert "generated_handle" in names
    assert "empty_bio" in names
    assert "missing_avatar" in names


def test_heuristics_flag_templated_posts() -> None:
    profile = _profile(
        username="promo",
        description="join my telegram for crypto signals",
        public_metrics={
            "followers_count": 100,
            "following_count": 5000,
            "tweet_count": 900,
        },
    )
    posts = [
        {
            "created_at": (datetime.now(UTC) - timedelta(hours=i * 3)).isoformat(),
            "text": "DM me for crypto signals now",
            "public_metrics": {
                "like_count": 0,
                "reply_count": 0,
                "retweet_count": 0,
                "quote_count": 0,
            },
        }
        for i in range(10)
    ]
    names = {s.name for s in run_heuristics(profile, posts)}
    assert "templated_posts" in names


@pytest.mark.asyncio
async def test_assess_account_combines_llm_verdict() -> None:
    """A confident LLM verdict on an otherwise clean profile stays 'unknown'.

    Neither signal may exclude an account on its own, so the LLM's verdict is
    recorded and surfaced, but the account is not filtered out.
    """
    llm = FakeLLM(
        [
            {
                "is_bot": True,
                "confidence": 0.95,
                "reasons": ["Posts the same crypto ad every hour"],
                "signals_present": ["repetitive posts"],
            }
        ]
    )
    assessment = await assess_account(_profile(), _posts(), llm=llm)
    assert assessment.llm_is_bot is True
    assert assessment.score > 0.5
    assert assessment.verdict != "likely_bot"
    assert "Posts the same crypto ad every hour" in assessment.reasons


@pytest.mark.asyncio
async def test_assess_account_excludes_when_both_signals_agree() -> None:
    """A bot with heuristic red flags and a confident LLM verdict is excluded."""
    bot_profile = _profile(
        username="cryptobot2024",
        description="crypto airdrop click link in bio",
        created_at="2024-01-15T00:00:00.000Z",
        public_metrics={
            "followers_count": 10,
            "following_count": 9000,
            "tweet_count": 800,
        },
    )
    bot_posts = [
        {
            "created_at": (datetime.now(UTC) - timedelta(hours=i * 3)).isoformat(),
            "text": "DM me for crypto signals now",
            "public_metrics": {
                "like_count": 0,
                "reply_count": 0,
                "retweet_count": 0,
                "quote_count": 0,
            },
        }
        for i in range(10)
    ]
    llm = FakeLLM(
        [
            {
                "is_bot": True,
                "confidence": 0.95,
                "reasons": ["Posts the same crypto ad every hour"],
                "signals_present": ["repetitive posts"],
            }
        ]
    )
    assessment = await assess_account(bot_profile, bot_posts, llm=llm)
    assert assessment.llm_is_bot is True
    assert assessment.score >= 0.6
    assert assessment.verdict == "likely_bot"


@pytest.mark.asyncio
async def test_assess_account_human_verdict() -> None:
    llm = FakeLLM(
        [{"is_bot": False, "confidence": 0.9, "reasons": [], "signals_present": []}]
    )
    assessment = await assess_account(_profile(), _posts(), llm=llm)
    assert assessment.verdict == "human"
    assert assessment.score < 0.3


@pytest.mark.asyncio
async def test_assess_account_survives_llm_failure() -> None:
    class BrokenLLM:
        async def converse_json(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("bedrock down")

    assessment = await assess_account(_profile(), _posts(), llm=BrokenLLM())
    assert assessment.llm_is_bot is None
    assert assessment.verdict == "human"


# ---------------------------------------------------------------------------
# End-to-end discovery with a mocked X API
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_run_discovery_filters_bots_and_extracts_key_info(
    db_session, auth_token, monkeypatch
) -> None:
    from app.models import DiscoveredAccount

    human = _profile("1", "alice_robotics")
    bot = _profile(
        "2",
        "cryptobot2024",
        description="crypto airdrop click link in bio",
        created_at="2024-01-15T00:00:00.000Z",
        public_metrics={
            "followers_count": 10,
            "following_count": 9000,
            "tweet_count": 800,
        },
    )

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [human, bot], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [human, bot]})
    )

    def timeline(request: Any) -> respx.Response:
        user_id = request.url.path.split("/")[-2]
        if user_id == "1":
            return httpx.Response(200, json={"data": _posts()})
        return httpx.Response(200, json={"data": []})

    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        side_effect=timeline
    )

    llm = FakeLLM(
        by_prompt={
            # Query planning
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": ['"robotics" -is:retweet'],
                "rationale": "r",
            },
            # Key information comes first so its marker is matched before the
            # shorter bot-verdict markers.
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": ["Engineer"],
                "specializations": ["legged robots"],
                "interests": ["controls"],
                "milestones": ["conference talk"],
                "future_plans": [],
                "evidence": [
                    {
                        "post_id": "post-0",
                        "url": "https://x.com/alice/status/post-0",
                        "quote": "Shipped a new IK solver today",
                        "supports": "milestones",
                    }
                ],
            },
            # Bot verdicts, keyed by the handle rendered into the prompt.
            "Handle: @cryptobot2024": {
                "is_bot": True,
                "confidence": 0.95,
                "reasons": ["automated postings"],
                "signals_present": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(max_accounts=10, max_candidates=10),
        )

    assert result.candidates_seen == 2
    assert result.bots_filtered == 1
    assert len(result.accounts) == 1
    assert result.accounts[0]["username"] == "alice_robotics"
    assert result.accounts[0]["bot"]["verdict"] == "human"
    assert result.accounts[0]["key_information"]["occupation"] == "Robotics Engineer"
    assert result.warnings == []

    stored = await db_session.get(DiscoveredAccount, "2")
    assert stored is not None
    assert stored.bot_verdict == "likely_bot"


@pytest.mark.asyncio
@respx.mock
async def test_run_discovery_with_seed_accounts(
    db_session, auth_token, monkeypatch
) -> None:
    seed = _profile("seed1", "carol_ai")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(6)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [seed], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )

    llm = FakeLLM(
        [
            {
                "keywords": ["ai researcher", "machine learning"],
                "exclude_keywords": ["crypto"],
                "rationale": "seeds are researchers",
            },
            {
                "user_queries": ["ai researcher"],
                "post_queries": ['"machine learning" -is:retweet'],
                "rationale": "r",
            },
            {"is_bot": False, "confidence": 0.9, "reasons": [], "signals_present": []},
            {
                "bio": "AI researcher",
                "occupation": "Researcher",
                "titles": [],
                "specializations": ["ML"],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        ]
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=["https://x.com/carol_ai"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )

    assert "ai researcher" in result.target_segment.keywords
    assert "machine learning" in result.target_segment.keywords
    # No caller-supplied segment, so the segment is purely LLM-derived.
    assert result.target_segment.source == "llm"
    assert result.accounts
    # The seed account the caller named must be in the results.
    assert any(account["x_user_id"] == "seed1" for account in result.accounts)


@pytest.mark.asyncio
@respx.mock
async def test_run_discovery_includes_the_seed_account(
    db_session, auth_token, monkeypatch
) -> None:
    """A named seed account is always returned, even when the search finds none.

    The seed is the account the caller explicitly asked about, so dropping it
    because the search happened to return nothing would lose the one account
    the request was about.
    """
    seed = _profile("seed42", "carol_ai")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(6)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    # The search returns nothing: only the seed can be returned.
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    llm = FakeLLM(
        [
            {
                "keywords": ["ai researcher"],
                "exclude_keywords": [],
                "rationale": "seeds are researchers",
            },
            {
                "user_queries": ["ai researcher"],
                "post_queries": ['"ai researcher" -is:retweet'],
                "rationale": "r",
            },
            {"is_bot": False, "confidence": 0.9, "reasons": [], "signals_present": []},
            {
                "bio": "AI researcher",
                "occupation": "Researcher",
                "titles": [],
                "specializations": ["ML"],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        ]
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=["https://x.com/carol_ai"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )

    assert [account["x_user_id"] for account in result.accounts] == ["seed42"]
    assert result.accounts[0]["origin"] == "seed"
    assert result.seeds_included == 1


@pytest.mark.asyncio
@respx.mock
async def test_run_discovery_returns_seeds_when_no_segment_can_be_derived(
    db_session, auth_token, monkeypatch
) -> None:
    """Without a usable segment the run still answers with the seed accounts."""
    seed = _profile("seed7", "dave_robotics")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(4)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    # The segment derivation returns no usable keywords.
    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": [],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=["https://x.com/dave_robotics"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )

    assert [account["x_user_id"] for account in result.accounts] == ["seed7"]
    assert any("seed" in warning.lower() for warning in result.warnings)
    # The search stage must not run without a segment.
    assert result.candidates_seen == 0


@pytest.mark.asyncio
@respx.mock
async def test_seed_accounts_are_truncated_to_max_accounts(
    db_session, auth_token, monkeypatch
) -> None:
    """max_accounts is the hard cap on the accounts returned.

    The seeds are truncated to the number of slots the searched accounts
    leave free, so the response never grows beyond the requested bound.
    """
    seeds = {
        "alice_seed": _profile("seed_a", "alice_seed"),
        "bob_seed": _profile("seed_b", "bob_seed"),
        "carol_seed": _profile("seed_c", "carol_seed"),
    }

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "data": [
                    seeds[request.url.params.get("usernames", "")]
                ]
            },
        )
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(3)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=[
                "https://x.com/alice_seed",
                "https://x.com/bob_seed",
                "https://x.com/carol_seed",
            ],
            limits=SearchLimits(max_accounts=2, max_candidates=5),
        )

    # Three seeds, but only two may be returned.
    assert len(result.accounts) == 2
    assert len(result.accounts) <= 2
    assert {account["x_user_id"] for account in result.accounts} == {
        "seed_a",
        "seed_b",
    }


@pytest.mark.asyncio
@respx.mock
async def test_seeds_fill_only_the_slots_the_search_left_free(
    db_session, auth_token, monkeypatch
) -> None:
    """The seeds are truncated to the slots the searched accounts leave free.

    Two accounts are found by the search and three seeds are supplied, so only
    one seed fits within the max_accounts limit of three.
    """
    seed = _profile("seed_a", "alice_seed")
    found = [_profile("found1", "frank_found"), _profile("found2", "grace_found")]

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(3)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": found, "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": found})
    )

    llm = FakeLLM(
        by_prompt={
            "Analyze these seed accounts": {
                "keywords": ["robotics engineer"],
                "exclude_keywords": [],
                "rationale": "the seed is a robotics engineer",
            },
            "Build a search plan": {
                "user_queries": ["robotics"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=["https://x.com/alice_seed"],
            limits=SearchLimits(max_accounts=3, max_candidates=5),
        )

    # Two found by the search plus one seed fills the limit of three.
    assert len(result.accounts) == 3
    origins = {account["x_user_id"]: account["origin"] for account in result.accounts}
    assert origins["found1"] == "search"
    assert origins["found2"] == "search"
    assert origins["seed_a"] == "seed"


@pytest.mark.asyncio
@respx.mock
async def test_a_suppressed_seed_account_is_not_returned(
    db_session, auth_token, monkeypatch
) -> None:
    """A seed goes through the same suppression check as a searched account."""
    from app.models import SuppressionEntry

    seed = _profile("seed9", "eve_robotics")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/by/username/[^/]+").mock(
        return_value=httpx.Response(200, json={"data": [seed]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(3)})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    db_session.add(SuppressionEntry(x_user_id="seed9", handle="eve_robotics"))
    await db_session.flush()

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.segment.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            seed_accounts=["https://x.com/eve_robotics"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )

    assert result.accounts == []


@pytest.mark.asyncio
async def test_run_discovery_without_segment_fails_gracefully(
    db_session, auth_token
) -> None:
    async with XClient(db_session) as client:
        result = await run_discovery(db_session, client)

    assert result.accounts == []
    assert any("target segment" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
@respx.mock
async def test_run_discovery_skips_suppressed_accounts(
    db_session, auth_token, monkeypatch
) -> None:
    """A suppressed account must never appear in discovery results."""
    from app.models import SuppressionEntry

    human = _profile("1", "alice_robotics")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [human], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [human]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts()})
    )

    db_session.add(SuppressionEntry(x_user_id="1", handle="alice_robotics"))
    await db_session.flush()

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(max_accounts=10, max_candidates=10),
        )

    assert result.accounts == []


def test_target_segment_merges_provided_keywords() -> None:
    from app.agents.segment import TargetSegment

    segment = TargetSegment(
        keywords=["robotics engineer", "embodied ai"], exclude_keywords=["crypto"]
    )
    merged = segment.merged_with(["robotics engineer", "vision systems"])

    assert merged.source == "merged"
    assert merged.keywords[0] == "robotics engineer"
    assert "vision systems" in merged.keywords
    assert merged.keywords.count("robotics engineer") == 1
    assert merged.exclude_keywords == ["crypto"]


# ---------------------------------------------------------------------------
# Capability-driven search mode
# ---------------------------------------------------------------------------
def test_resolve_search_mode_prefers_full_archive_when_available() -> None:
    from app.agents.search import _resolve_search_mode
    from app.x.capabilities import Capabilities

    caps = Capabilities(full_archive_search=True, recent_search=True)
    mode, note = _resolve_search_mode(caps)

    assert mode == "full_archive"
    assert note == ""


def test_resolve_search_mode_falls_back_to_recent() -> None:
    from app.agents.search import _resolve_search_mode
    from app.x.capabilities import Capabilities

    caps = Capabilities(full_archive_search=False, recent_search=True)
    mode, note = _resolve_search_mode(caps)

    assert mode == "recent"
    assert note == ""


def test_resolve_search_mode_disables_post_search_when_nothing_is_available() -> None:
    """Neither search endpoint available must not fall back to a doomed call."""
    from app.agents.search import _resolve_search_mode
    from app.x.capabilities import Capabilities

    caps = Capabilities(full_archive_search=False, recent_search=False)
    mode, note = _resolve_search_mode(caps)

    assert mode == "none"
    assert "unavailable" in note


@pytest.mark.asyncio
@respx.mock
async def test_discovery_skips_tweet_search_when_the_probe_refused_it(
    db_session, auth_token, monkeypatch
) -> None:
    """A refused tweet-search endpoint must not be called again.

    The capability probe already learned that ``/2/tweets/search/recent`` is
    unavailable. Re-calling it during the search stage produced a stream of
    identical 400s and silently yielded zero post-derived candidates.
    """
    from app.x.capabilities import capability_probe

    profile = _profile("1", "alice_robotics")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    recent_route = respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    all_route = respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [profile], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [profile]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(3)})
    )

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": ['"robotics" -is:retweet'],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    capability_probe.invalidate()

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(max_accounts=10, max_candidates=10),
        )

    # The probe called each endpoint once; the search stage must not add more.
    assert recent_route.call_count == 1
    assert all_route.call_count == 1

    # User-search candidates still come through, and the limitation is reported.
    assert [account["username"] for account in result.accounts] == ["alice_robotics"]
    assert any("unavailable" in warning for warning in result.warnings)


@pytest.mark.asyncio
@respx.mock
async def test_small_candidate_budget_does_not_over_read(
    db_session, auth_token, monkeypatch
) -> None:
    """A small ``max_candidates`` must not pull whole pages of results.

    The X search endpoints always return up to ``max_results`` items, so a
    request larger than the remaining budget reads far more than the caller
    asked for and burns the X API budget.
    """
    user_search = respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": _profiles(50), "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(200, json={"data": _posts(100), "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/users").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "data": [
                    p
                    for p in _profiles(50)
                    if p["id"] in request.url.params.get("ids", "").split(",")
                ]
            },
        )
    )
    timeline = respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(100)})
    )

    from app.x.capabilities import capability_probe

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": ['"robotics" -is:retweet'],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": ["Engineer"],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    capability_probe.invalidate()

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(
                max_accounts=1,
                max_candidates=1,
                max_posts_per_account=0,
            ),
        )

    # The page size must respect the remaining candidate budget.
    assert int(user_search.calls[0].request.url.params["max_results"]) <= 1
    # No post was requested for evidence, so no timeline read happens at all.
    assert timeline.call_count == 0
    # Only the budgeted candidates are considered.
    assert result.candidates_seen <= 1


@pytest.mark.asyncio
@respx.mock
async def test_zero_posts_per_account_skips_the_timeline_read(
    db_session, auth_token, monkeypatch
) -> None:
    """``max_posts_per_account=0`` must not read a single post."""
    from app.x.capabilities import capability_probe

    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": _profiles(1), "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": _profiles(1)})
    )
    timeline = respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(20)})
    )

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    capability_probe.invalidate()

    async with XClient(db_session) as client:
        result = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(
                max_accounts=1,
                max_candidates=5,
                max_posts_per_account=0,
            ),
        )

    assert timeline.call_count == 0
    assert result.estimated_cost_usd < 0.1


@pytest.mark.asyncio
@respx.mock
async def test_a_new_search_purges_the_previous_runs(
    db_session, auth_token, monkeypatch
) -> None:
    """Every new search starts from an empty discovery history.

    The previous runs and the accounts they produced are removed, so only the
    latest run and its accounts remain visible.
    """
    from sqlalchemy import select

    from app.models import DiscoveredAccount, DiscoveryRun

    found = _profile("1", "alice_robotics")

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [found], "meta": {}})
    )
    respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [found]})
    )
    respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": _posts(3)})
    )

    llm = FakeLLM(
        by_prompt={
            "Build a search plan": {
                "user_queries": ["robotics engineer"],
                "post_queries": [],
                "rationale": "r",
            },
            "Extract key information about this person": {
                "bio": "Robotics engineer",
                "occupation": "Robotics Engineer",
                "titles": [],
                "specializations": [],
                "interests": [],
                "milestones": [],
                "future_plans": [],
                "evidence": [],
            },
            "Handle: @alice_robotics": {
                "is_bot": False,
                "confidence": 0.95,
                "reasons": [],
                "signals_present": [],
            },
        }
    )
    monkeypatch.setattr("app.agents.queries.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.profile.get_llm", lambda: llm)
    monkeypatch.setattr("app.agents.botdetect.get_llm", lambda: llm)

    capability_probe.invalidate()

    async with XClient(db_session) as client:
        first = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )
        second = await run_discovery(
            db_session,
            client,
            target_segment=["robotics engineer"],
            limits=SearchLimits(max_accounts=5, max_candidates=5),
        )

    # Only the second run survives.
    runs = await db_session.execute(select(DiscoveryRun))
    remaining_runs = runs.scalars().all()
    assert len(remaining_runs) == 1
    assert remaining_runs[0].id == second.run_id
    assert remaining_runs[0].id != first.run_id

    # The accounts of the first run are gone with it.
    accounts = await db_session.execute(select(DiscoveredAccount))
    remaining_accounts = accounts.scalars().all()
    assert len(remaining_accounts) == len(second.accounts)
    assert all(
        account.source_run_id == second.run_id for account in remaining_accounts
    )
