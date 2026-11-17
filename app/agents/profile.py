"""Stage 4 of discovery: extract a person's key information.

The summary is what a human reviewer reads before approving an outreach message,
so it must be grounded in real content and must carry its own citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from app.agents.botdetect import render_posts, render_profile
from app.llm.bedrock import BedrockLLM, get_llm
from app.llm.prompts import (
    ANTI_FABRICATION,
    KEY_INFO_SCHEMA,
    KEY_INFO_SYSTEM,
    KEY_INFO_USER,
)

log = structlog.get_logger(__name__)


@dataclass
class Evidence:
    """One cited piece of content."""

    post_id: str
    url: str
    quote: str
    supports: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "url": self.url,
            "quote": self.quote,
            "supports": self.supports,
        }


@dataclass
class KeyInformation:
    """The structured summary of a person."""

    bio: str = ""
    occupation: str = ""
    titles: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    milestones: list[str] = field(default_factory=list)
    future_plans: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "bio": self.bio,
            "occupation": self.occupation,
            "titles": self.titles,
            "specializations": self.specializations,
            "interests": self.interests,
            "milestones": self.milestones,
            "future_plans": self.future_plans,
            "evidence": [e.to_payload() for e in self.evidence],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> KeyInformation:
        return cls(
            bio=str(payload.get("bio") or ""),
            occupation=str(payload.get("occupation") or ""),
            titles=[str(t) for t in (payload.get("titles") or [])],
            specializations=[str(s) for s in (payload.get("specializations") or [])],
            interests=[str(i) for i in (payload.get("interests") or [])],
            milestones=[str(m) for m in (payload.get("milestones") or [])],
            future_plans=[str(f) for f in (payload.get("future_plans") or [])],
            evidence=[
                Evidence(
                    post_id=str(e.get("post_id") or ""),
                    url=str(e.get("url") or ""),
                    quote=str(e.get("quote") or ""),
                    supports=str(e.get("supports") or ""),
                )
                for e in (payload.get("evidence") or [])
                if isinstance(e, dict)
            ],
        )


async def extract_key_information(
    profile: dict[str, Any],
    posts: list[dict[str, Any]],
    llm: BedrockLLM | None = None,
) -> KeyInformation:
    """Extract a grounded key-information summary for one account."""
    llm = llm or get_llm()

    try:
        result = await llm.converse_json(
            system=KEY_INFO_SYSTEM,
            user=KEY_INFO_USER.format(
                profile=render_profile(profile),
                posts=render_posts(posts, limit=20),
                anti_fabrication=ANTI_FABRICATION,
            ),
            schema=KEY_INFO_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("key_info_failed", error=str(exc))
        return KeyInformation(
            bio=str(profile.get("description") or ""),
        )

    info = KeyInformation.from_payload(result)
    info.evidence = _keep_known_evidence(info.evidence, posts)
    return info


def _keep_known_evidence(
    evidence: list[Evidence], posts: list[dict[str, Any]]
) -> list[Evidence]:
    """Drop citations whose post id is not among the posts we actually fetched.

    This guards against the model citing content it was never shown.
    """
    known_ids = {str(p.get("id")) for p in posts if p.get("id")}
    if not known_ids:
        return evidence
    return [e for e in evidence if not e.post_id or e.post_id in known_ids]


def render_key_information(info: KeyInformation) -> str:
    """Render the summary for a prompt."""
    lines = []
    if info.occupation:
        lines.append(f"Occupation: {info.occupation}")
    if info.titles:
        lines.append(f"Titles: {', '.join(info.titles)}")
    if info.specializations:
        lines.append(f"Specializations: {', '.join(info.specializations)}")
    if info.interests:
        lines.append(f"Interests: {', '.join(info.interests)}")
    if info.milestones:
        lines.append(f"Milestones: {', '.join(info.milestones)}")
    if info.future_plans:
        lines.append(f"Future plans: {', '.join(info.future_plans)}")
    if info.bio:
        lines.append(f"Bio: {info.bio}")
    if not lines:
        return "(no key information extracted)"
    return "\n".join(lines)


def render_evidence(
    evidence: list[Evidence], posts: list[dict[str, Any]] | None = None
) -> str:
    """Render the cited evidence for a prompt."""
    if not evidence:
        return "(no cited evidence)"
    lines = []
    for item in evidence[:10]:
        lines.append(f"- {item.quote}")
        if item.url:
            lines.append(f"  {item.url}")
    return "\n".join(lines)
