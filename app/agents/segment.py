"""Stage 1 of discovery: turn seed accounts into a target segment.

Seed accounts are resolved to profiles and their posts are read newest-first.
The LLM then distils those into a keyword segment that drives the X search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from app.llm.bedrock import BedrockLLM, get_llm
from app.llm.prompts import (
    ANTI_FABRICATION,
    SEED_SEGMENT_SCHEMA,
    SEED_SEGMENT_SYSTEM,
    SEED_SEGMENT_USER,
)
from app.x.client import XClient
from app.x.errors import x_error_fields

log = structlog.get_logger(__name__)

#: How many posts to read from each seed account.
SEED_POSTS_PER_ACCOUNT = 20


@dataclass
class TargetSegment:
    """The derived search segment."""

    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "llm"  # "llm", "provided" or "merged"

    def merged_with(self, provided: list[str]) -> TargetSegment:
        """Union the caller-provided keywords with the derived ones."""
        combined: list[str] = []
        seen: set[str] = set()
        for keyword in [*provided, *self.keywords]:
            cleaned = keyword.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                combined.append(cleaned)
        return TargetSegment(
            keywords=combined,
            exclude_keywords=list(self.exclude_keywords),
            rationale=self.rationale,
            source="merged" if provided else self.source,
        )


@dataclass
class SeedProfile:
    """A resolved seed account and the content read from it."""

    username: str
    x_user_id: str
    display_name: str = ""
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    posts: list[dict[str, Any]] = field(default_factory=list)
    resolved: bool = True
    error: str = ""

    def to_evidence_text(self) -> str:
        """Render this profile for the prompt."""
        lines = [
            f"Handle: @{self.username}",
            f"Name: {self.display_name}",
            f"Bio: {self.bio or '(empty)'}",
            f"Followers: {self.followers_count}, Following: {self.following_count}",
        ]
        if self.posts:
            lines.append("Recent posts (newest first):")
            for post in self.posts:
                text = (post.get("text") or "").replace("\n", " ").strip()
                if text:
                    lines.append(f"  - {text}")
        else:
            lines.append("Recent posts: none available")
        return "\n".join(lines)


def parse_seed_handle(url: str) -> str | None:
    """Extract a handle from an X profile URL.

    Accepts ``https://x.com/handle``, ``https://twitter.com/handle``,
    ``@handle`` and a bare ``handle``.
    """
    candidate = url.strip()
    if not candidate:
        return None

    if candidate.startswith("@"):
        candidate = candidate[1:]

    lowered = candidate.lower()
    prefixes = (
        "https://www.twitter.com/",
        "https://www.x.com/",
        "https://twitter.com/",
        "https://x.com/",
        "http://twitter.com/",
        "http://x.com/",
        "www.twitter.com/",
        "www.x.com/",
        "twitter.com/",
        "x.com/",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    else:
        if "://" in candidate or "/" in candidate:
            return None

    handle = candidate.split("/")[0].split("?")[0].strip()
    if not handle or not handle.replace("_", "").isalnum():
        return None
    return handle


async def collect_seed_profiles(
    client: XClient,
    seed_urls: list[str],
    *,
    posts_per_account: int = SEED_POSTS_PER_ACCOUNT,
) -> list[SeedProfile]:
    """Resolve each seed URL to a profile and read its posts (newest first)."""
    profiles: list[SeedProfile] = []
    seen: set[str] = set()

    for url in seed_urls:
        handle = parse_seed_handle(url)
        if handle is None:
            profiles.append(
                SeedProfile(
                    username=url,
                    x_user_id="",
                    resolved=False,
                    error=f"Could not parse an X handle from {url!r}",
                )
            )
            continue

        key = handle.lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            user = await client.get_user_by_username(handle)
        except Exception as exc:  # noqa: BLE001 - upstream failures become seed errors
            log.warning("seed_lookup_failed", handle=handle, **x_error_fields(exc))
            profiles.append(
                SeedProfile(
                    username=handle,
                    x_user_id="",
                    resolved=False,
                    error=str(exc),
                )
            )
            continue

        if user is None:
            profiles.append(
                SeedProfile(
                    username=handle,
                    x_user_id="",
                    resolved=False,
                    error="Account not found on X",
                )
            )
            continue

        metrics = user.get("public_metrics") or {}
        profile = SeedProfile(
            username=user.get("username") or handle,
            x_user_id=str(user.get("id") or ""),
            display_name=user.get("name") or "",
            bio=user.get("description") or "",
            followers_count=int(metrics.get("followers_count") or 0),
            following_count=int(metrics.get("following_count") or 0),
        )

        try:
            tweets = await client.get_user_tweets(
                profile.x_user_id, max_results=posts_per_account
            )
            profile.posts = list(tweets.get("data") or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("seed_posts_failed", handle=handle, **x_error_fields(exc))
            profile.error = f"Could not read posts: {exc}"

        profiles.append(profile)

    return profiles


async def derive_target_segment(
    seed_profiles: list[SeedProfile],
    llm: BedrockLLM | None = None,
) -> TargetSegment:
    """Ask the LLM to derive a target segment from the seed profiles."""
    llm = llm or get_llm()
    usable = [p for p in seed_profiles if p.resolved]
    if not usable:
        return TargetSegment(rationale="No seed account could be resolved.")

    seed_text = "\n\n".join(p.to_evidence_text() for p in usable)

    try:
        result = await llm.converse_json(
            system=SEED_SEGMENT_SYSTEM,
            user=SEED_SEGMENT_USER.format(
                seed_profiles=seed_text,
                anti_fabrication=ANTI_FABRICATION,
            ),
            schema=SEED_SEGMENT_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("segment_derivation_failed", error=str(exc))
        # Fall back to the raw bio terms so a discovery run can still proceed.
        return TargetSegment(
            keywords=_fallback_keywords(usable),
            rationale="LLM derivation failed; falling back to raw profile terms.",
        )

    keywords = [str(k).strip() for k in (result.get("keywords") or []) if str(k).strip()]
    excludes = [str(k).strip() for k in (result.get("exclude_keywords") or []) if str(k).strip()]

    return TargetSegment(
        keywords=keywords or _fallback_keywords(usable),
        exclude_keywords=excludes,
        rationale=str(result.get("rationale") or ""),
    )


def _fallback_keywords(profiles: list[SeedProfile]) -> list[str]:
    """Derive crude keywords from bios when the LLM is unavailable."""
    keywords: list[str] = []
    for profile in profiles:
        bio = profile.bio.replace("\n", " ").strip()
        if not bio:
            continue
        # Take the first short phrase of the bio as a crude role descriptor.
        for chunk in bio.split(","):
            cleaned = chunk.strip().lstrip("|").strip()
            if 3 <= len(cleaned) <= 50:
                keywords.append(cleaned)
            if len(keywords) >= 12:
                break
        if len(keywords) >= 12:
            break
    return keywords[:12]
