"""Hybrid bot detection.

The deterministic half implements every red flag listed in
``build-docs/bot-detection.md``; the LLM half judges the profile and posts for
signals that are hard to express as a rule. The two are combined into one score
in the range 0.0 - 1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import structlog

from app.llm.bedrock import BedrockLLM, get_llm
from app.llm.prompts import (
    ANTI_FABRICATION,
    BOT_DETECTION_SCHEMA,
    BOT_DETECTION_SYSTEM,
    BOT_DETECTION_USER,
)
from app.models import BotVerdict

log = structlog.get_logger(__name__)

#: Accounts created after this date are considered "recently created" (per the doc).
RECENT_ACCOUNT_CUTOFF = datetime(2023, 10, 1, tzinfo=UTC)

#: A handle like "John4821" - a first name followed by digits.
_GENERATED_HANDLE = re.compile(r"^[A-Za-z]{2,15}\d{3,}$")
#: Bio terms that indicate a link-only or promotion-heavy bio.
_SPAM_BIO_TERMS = (
    "crypto",
    "casino",
    "betting",
    "forex",
    "nft",
    "airdrop",
    "onlyfans",
    "adult",
    "escort",
    "viagra",
    "giveaway",
    "click link",
    "link in bio",
    "dm me",
    "join my",
    "telegram",
    "whatsapp",
)
#: Text that is almost entirely a URL.
_URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$", re.IGNORECASE)


@dataclass
class HeuristicSignal:
    """One red flag detected by a deterministic rule."""

    name: str
    weight: float
    detail: str

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "detail": self.detail}


@dataclass
class BotAssessment:
    """The combined verdict for one account."""

    score: float = 0.0
    verdict: str = BotVerdict.UNKNOWN.value
    reasons: list[str] = field(default_factory=list)
    heuristic_signals: list[HeuristicSignal] = field(default_factory=list)
    llm_is_bot: bool | None = None
    llm_confidence: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "llm_is_bot": self.llm_is_bot,
            "llm_confidence": self.llm_confidence,
            "heuristic_signals": [s.to_payload() for s in self.heuristic_signals],
        }


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


# ---------------------------------------------------------------------------
# Deterministic heuristics
# ---------------------------------------------------------------------------
def check_follow_ratio(profile: dict[str, Any]) -> HeuristicSignal | None:
    """Flag accounts that follow thousands but have few followers back."""
    metrics = profile.get("public_metrics") or {}
    followers = int(metrics.get("followers_count") or 0)
    following = int(metrics.get("following_count") or 0)
    if following < 500:
        return None
    if followers == 0:
        return HeuristicSignal(
            "lopsided_follow_ratio",
            0.30,
            f"Follows {following} accounts and has {followers} followers.",
        )
    ratio = following / max(followers, 1)
    if ratio >= 20:
        return HeuristicSignal(
            "lopsided_follow_ratio",
            0.30,
            f"Follows {following} accounts with only {followers} followers "
            f"(ratio {ratio:.0f}:1).",
        )
    if ratio >= 10:
        return HeuristicSignal(
            "mild_follow_ratio",
            0.12,
            f"Follows {following} accounts with {followers} followers "
            f"(ratio {ratio:.0f}:1).",
        )
    return None


def check_bio(profile: dict[str, Any]) -> list[HeuristicSignal]:
    """Flag an empty bio or a single-issue/link-only bio."""
    signals: list[HeuristicSignal] = []
    bio = (profile.get("description") or "").strip()
    handle = (profile.get("username") or "").strip()

    # The handle check is independent of the bio, so it runs first.
    if handle and _GENERATED_HANDLE.match(handle):
        signals.append(
            HeuristicSignal(
                "generated_handle",
                0.22,
                f"The handle @{handle} looks like a first name plus random digits.",
            )
        )

    if not bio:
        signals.append(
            HeuristicSignal("empty_bio", 0.20, "The profile has an empty bio.")
        )
        return signals

    lowered = bio.lower()
    matched = [term for term in _SPAM_BIO_TERMS if term in lowered]
    if matched:
        signals.append(
            HeuristicSignal(
                "single_issue_bio",
                0.28,
                f"The bio promotes a single topic or link ({', '.join(matched[:3])}).",
            )
        )

    if _URL_ONLY.match(bio):
        signals.append(
            HeuristicSignal("link_only_bio", 0.22, "The bio contains only a link.")
        )
    elif len(bio) < 25:
        signals.append(
            HeuristicSignal(
                "bare_bio",
                0.10,
                f"The bio is very short ({len(bio)} characters).",
            )
        )

    return signals


def check_avatar(profile: dict[str, Any]) -> HeuristicSignal | None:
    """Flag a blank or default avatar."""
    avatar = (profile.get("profile_image_url") or "").strip()
    if not avatar:
        return HeuristicSignal(
            "missing_avatar", 0.15, "The account has no profile image."
        )
    lowered = avatar.lower()
    if any(
        marker in lowered
        for marker in ("default", "placeholder", "default_profile")
    ):
        return HeuristicSignal(
            "default_avatar",
            0.18,
            "The profile image is X's default avatar.",
        )
    return None


def check_account_age(profile: dict[str, Any], post_count: int) -> HeuristicSignal | None:
    """Flag a brand-new account that already posts at volume."""
    created = _parse_datetime(profile.get("created_at"))
    if created is None:
        return None
    if created < RECENT_ACCOUNT_CUTOFF:
        return None
    if post_count < 20:
        return None
    return HeuristicSignal(
        "recent_account_high_volume",
        0.22,
        f"The account was created on {created.date().isoformat()} and already has "
        f"{post_count} posts.",
    )


def check_posting_cadence(posts: list[dict[str, Any]]) -> HeuristicSignal | None:
    """Flag round-the-clock posting with no human sleep gaps.

    Posting across many hours of the day is normal for an active professional.
    The red flag is sustained posting through the small hours, so this requires a
    high density of overnight posts across a multi-day window.
    """
    timestamps = [
        ts
        for ts in (_parse_datetime(p.get("created_at")) for p in posts)
        if ts is not None
    ]
    if len(timestamps) < 12:
        return None

    timestamps.sort()
    span_days = (timestamps[-1] - timestamps[0]).days
    if span_days < 3:
        return None

    # Sleep hours: 00:00-05:59 UTC.
    overnight = [ts for ts in timestamps if ts.astimezone(UTC).hour < 6]
    overnight_ratio = len(overnight) / len(timestamps)

    # A human typically posts a small fraction of their posts overnight.
    if overnight_ratio >= 0.30 and len(overnight) >= 5:
        return HeuristicSignal(
            "round_the_clock_activity",
            0.20,
            f"{len(overnight)} of {len(timestamps)} posts were published between "
            "00:00 and 06:00 UTC over a "
            f"{span_days}-day window, with no clear rest period.",
        )
    return None


def check_repetitive_content(
    posts: list[dict[str, Any]],
) -> HeuristicSignal | None:
    """Flag identical or near-identical repeated posts.

    Two patterns matter: a near-identical pair, and a template repeated across
    most of the recent posts (the scripted-reply pattern from the doc).
    """
    texts = [
        (p.get("text") or "").strip().lower()
        for p in posts
        if (p.get("text") or "").strip()
    ]
    if len(texts) < 5:
        return None

    best_ratio = 0.0
    best_pair: tuple[str, str] = ("", "")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (texts[i], texts[j])

    # A template repeated across most posts is stronger evidence than a single pair.
    normalized = [_normalize_template(text) for text in texts]
    template_counts: dict[str, int] = {}
    for norm in normalized:
        template_counts[norm] = template_counts.get(norm, 0) + 1
    most_common_template, template_count = max(
        template_counts.items(), key=lambda item: item[1]
    )
    if template_count >= max(4, len(texts) * 0.5):
        return HeuristicSignal(
            "templated_posts",
            0.30,
            f"{template_count} of {len(texts)} recent posts share the same template "
            f"({most_common_template[:60]!r}).",
        )

    if best_ratio >= 0.9:
        return HeuristicSignal(
            "repetitive_content",
            0.30,
            f"Two recent posts are {best_ratio:.0%} identical: "
            f"{best_pair[0][:60]!r}.",
        )
    if best_ratio >= 0.75:
        return HeuristicSignal(
            "near_repetitive_content",
            0.15,
            f"Two recent posts are {best_ratio:.0%} similar: {best_pair[0][:60]!r}.",
        )
    return None


def _normalize_template(text: str) -> str:
    """Strip digits, URLs and punctuation so template variants collapse together."""
    without_urls = re.sub(r"https?://\S+", "", text)
    without_numbers = re.sub(r"\d+", "#", without_urls)
    return re.sub(r"\s+", " ", without_numbers).strip()


def check_engagement(
    profile: dict[str, Any], posts: list[dict[str, Any]]
) -> HeuristicSignal | None:
    """Flag a large following with essentially no engagement on recent posts."""
    metrics = profile.get("public_metrics") or {}
    followers = int(metrics.get("followers_count") or 0)
    if followers < 1000 or not posts:
        return None

    engagements = 0
    for post in posts:
        post_metrics = post.get("public_metrics") or {}
        engagements += (
            int(post_metrics.get("like_count") or 0)
            + int(post_metrics.get("reply_count") or 0)
            + int(post_metrics.get("retweet_count") or 0)
            + int(post_metrics.get("quote_count") or 0)
        )

    per_post = engagements / len(posts)
    if per_post >= 1.0:
        return None

    return HeuristicSignal(
        "high_followers_no_engagement",
        0.22,
        f"Has {followers} followers but only {engagements} total engagements "
        f"({per_post:.2f} per post) across {len(posts)} recent posts.",
    )


def run_heuristics(
    profile: dict[str, Any], posts: list[dict[str, Any]]
) -> list[HeuristicSignal]:
    """Run every deterministic rule from ``bot-detection.md``."""
    metrics = profile.get("public_metrics") or {}
    signals: list[HeuristicSignal] = []

    follow = check_follow_ratio(profile)
    if follow:
        signals.append(follow)

    signals.extend(check_bio(profile))

    avatar = check_avatar(profile)
    if avatar:
        signals.append(avatar)

    age = check_account_age(profile, int(metrics.get("tweet_count") or len(posts)))
    if age:
        signals.append(age)

    cadence = check_posting_cadence(posts)
    if cadence:
        signals.append(cadence)

    repetitive = check_repetitive_content(posts)
    if repetitive:
        signals.append(repetitive)

    engagement = check_engagement(profile, posts)
    if engagement:
        signals.append(engagement)

    return signals


# ---------------------------------------------------------------------------
# LLM judgment
# ---------------------------------------------------------------------------
def render_profile(profile: dict[str, Any]) -> str:
    """Render a profile for the LLM prompt."""
    metrics = profile.get("public_metrics") or {}
    created = _parse_datetime(profile.get("created_at"))
    lines = [
        f"Handle: @{profile.get('username', '')}",
        f"Name: {profile.get('name', '')}",
        f"Bio: {profile.get('description') or '(empty)'}",
        f"Followers: {metrics.get('followers_count', 0)}",
        f"Following: {metrics.get('following_count', 0)}",
        f"Total posts: {metrics.get('tweet_count', 0)}",
        f"Created: {created.date().isoformat() if created else 'unknown'}",
        f"Verified: {profile.get('verified', False)}",
        f"Location: {profile.get('location') or '(none)'}",
    ]
    return "\n".join(lines)


def render_posts(posts: list[dict[str, Any]], limit: int = 20) -> str:
    """Render recent posts for the LLM prompt, newest first."""
    if not posts:
        return "(no posts available)"
    lines = []
    for post in posts[:limit]:
        created = _parse_datetime(post.get("created_at"))
        stamp = created.isoformat() if created else "unknown time"
        text = (post.get("text") or "").replace("\n", " ").strip()
        lines.append(f"- [{stamp}] {text}")
    return "\n".join(lines)


async def llm_bot_verdict(
    profile: dict[str, Any],
    posts: list[dict[str, Any]],
    llm: BedrockLLM | None = None,
) -> tuple[bool | None, float | None, list[str]]:
    """Ask the LLM whether the account is a bot.

    Returns ``(is_bot, confidence, reasons)``. On failure the verdict is unknown
    and the deterministic half of the assessment stands on its own.
    """
    llm = llm or get_llm()
    try:
        result = await llm.converse_json(
            system=BOT_DETECTION_SYSTEM,
            user=BOT_DETECTION_USER.format(
                profile=render_profile(profile),
                posts=render_posts(posts),
                max_posts=20,
                anti_fabrication=ANTI_FABRICATION,
            ),
            schema=BOT_DETECTION_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("bot_llm_verdict_failed", error=str(exc))
        return None, None, []

    is_bot = bool(result.get("is_bot"))
    raw_confidence = result.get("confidence")
    try:
        confidence = float(raw_confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        confidence = 0.5
    reasons = [str(r) for r in (result.get("reasons") or [])]
    return is_bot, max(0.0, min(1.0, confidence)), reasons


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------
def combine(
    signals: list[HeuristicSignal],
    llm_is_bot: bool | None,
    llm_confidence: float | None,
    llm_reasons: list[str],
    threshold: float = 0.6,
) -> BotAssessment:
    """Combine the heuristic and LLM verdicts into a single score and verdict.

    The two signals are combined so that neither can single-handedly exclude an
    account, but a confident LLM verdict combined with any heuristic signal is
    enough to cross the threshold:

    * heuristics alone cap at 0.45 (below any default threshold),
    * the LLM alone caps at 0.55 (also below the threshold),
    * a confident LLM verdict plus at least one heuristic signal reaches 0.6+.
    """
    heuristic_score = min(1.0, sum(s.weight for s in signals))
    reasons = [s.detail for s in signals]

    llm_score = 0.0
    if llm_is_bot is not None and llm_confidence is not None:
        llm_score = llm_confidence if llm_is_bot else 0.0
        reasons.extend(llm_reasons)

    if llm_score > 0 and heuristic_score > 0:
        # Both agree: take the stronger signal.
        combined = max(heuristic_score, llm_score)
    else:
        # Only one side fired: cap it so it cannot exclude an account alone.
        combined = 0.45 * heuristic_score if heuristic_score else 0.55 * llm_score

    if combined >= threshold:
        verdict = BotVerdict.LIKELY_BOT.value
    elif combined >= threshold * 0.5:
        verdict = BotVerdict.UNKNOWN.value
    else:
        verdict = BotVerdict.HUMAN.value

    return BotAssessment(
        score=combined,
        verdict=verdict,
        reasons=reasons,
        heuristic_signals=signals,
        llm_is_bot=llm_is_bot,
        llm_confidence=llm_confidence,
    )


async def assess_account(
    profile: dict[str, Any],
    posts: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
    llm: BedrockLLM | None = None,
) -> BotAssessment:
    """Run the full hybrid assessment for one account."""
    signals = run_heuristics(profile, posts)
    llm_is_bot, llm_confidence, llm_reasons = await llm_bot_verdict(profile, posts, llm)
    return combine(
        signals,
        llm_is_bot,
        llm_confidence,
        llm_reasons,
        threshold=threshold,
    )
