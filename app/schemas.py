"""Pydantic request and response models for the public API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    """Base model with the shared configuration."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
class DiscoveryRequest(ApiModel):
    """Search for accounts matching a target segment and/or seed accounts."""

    target_segment: list[str] = Field(
        default_factory=list,
        description="Keywords describing the people to find, e.g. ['robotics engineer'].",
    )
    seed_accounts: list[str] = Field(
        default_factory=list,
        description=(
            "X profile URLs of seed accounts, e.g. ['https://x.com/someone']. "
            "Their bios and posts are analysed to derive a target segment."
        ),
    )
    max_accounts: int = Field(default=40, ge=1, le=200)
    max_candidates: int = Field(default=150, ge=1, le=500)
    max_posts_per_account: int = Field(default=20, ge=0, le=100)
    include_bots: bool = Field(
        default=False,
        description="Return likely bots in the results instead of filtering them out.",
    )
    bot_score_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Accounts scoring at or above this are treated as bots.",
    )

    @field_validator("target_segment", "seed_accounts")
    @classmethod
    def _strip_and_dedupe(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower())
                cleaned.append(item)
        return cleaned


class EvidenceItem(ApiModel):
    post_id: str = ""
    url: str = ""
    quote: str = ""
    supports: str = ""


class KeyInformation(ApiModel):
    bio: str = ""
    occupation: str = ""
    titles: list[str] = Field(default_factory=list)
    specializations: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    future_plans: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class BotAssessment(ApiModel):
    verdict: str = "unknown"
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    llm_is_bot: bool | None = None
    llm_confidence: float | None = None
    heuristic_signals: list[dict[str, Any]] = Field(default_factory=list)


class DiscoveredAccount(ApiModel):
    x_user_id: str
    username: str
    display_name: str | None = None
    profile_url: str
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    created_at: str | None = None
    bot: BotAssessment = Field(default_factory=BotAssessment)
    key_information: KeyInformation = Field(default_factory=KeyInformation)
    origin: str = Field(
        default="search",
        description=(
            '"seed" when the account was supplied by the caller as a seed, '
            '"search" when the X search found it.'
        ),
    )


class QueryUsed(ApiModel):
    endpoint: str
    query: str


class DiscoveryStats(ApiModel):
    candidates: int = 0
    bots_filtered: int = 0
    returned: int = 0
    api_calls: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    seeds_supplied: int = 0
    seeds_resolved: int = 0
    seeds_included: int = 0
    seeds_dropped: int = 0


class DiscoveryResponse(ApiModel):
    run_id: uuid.UUID
    target_segment: list[str] = Field(default_factory=list)
    segment_rationale: str = ""
    queries_used: list[QueryUsed] = Field(default_factory=list)
    accounts: list[DiscoveredAccount] = Field(default_factory=list)
    stats: DiscoveryStats = Field(default_factory=DiscoveryStats)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DiscoveryAccepted(ApiModel):
    """The acknowledgement of a search that is still running.

    A discovery run waits on the X API and on several LLM inferences, which
    takes longer than an API Gateway integration may hold a connection open.
    The search is therefore started in the background and the caller polls
    ``poll_url`` until the run leaves the running state.
    """

    run_id: uuid.UUID
    status: str = "RUNNING"
    poll_url: str = ""


class DiscoveryRunSummary(ApiModel):
    id: uuid.UUID
    input_type: str
    target_segment: list[str] = Field(default_factory=list)
    seed_accounts: list[str] = Field(default_factory=list)
    status: str
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AccountRecord(ApiModel):
    x_user_id: str
    username: str
    display_name: str | None = None
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweets_count: int = 0
    verified: bool = False
    created_at: datetime | None = None
    bot_verdict: str = "unknown"
    bot_score: float = 0.0
    bot_reasons: list[str] = Field(default_factory=list)
    key_information: KeyInformation = Field(default_factory=KeyInformation)
    segment_tags: list[str] = Field(default_factory=list)
    source_run_id: uuid.UUID | None = None
    last_seen_at: datetime


# ---------------------------------------------------------------------------
# Outreach
# ---------------------------------------------------------------------------
class DraftRequest(ApiModel):
    x_user_id: str = Field(description="The X user id of the recipient.")
    goal: str = Field(
        default="",
        description="What the message should achieve, e.g. 'Invite them to a robotics meetup'.",
    )
    tone: str = Field(default="concise and warm")
    is_reply: bool = Field(
        default=False, description="Mark this draft as a reply to an inbound message."
    )


class DraftVersion(ApiModel):
    version: int
    text: str
    citations: list[EvidenceItem] = Field(default_factory=list)
    personalization_notes: str = ""
    feedback: str = ""
    created_at: datetime | None = None


class DraftResponse(ApiModel):
    id: uuid.UUID
    x_user_id: str
    handle: str | None = None
    status: str
    goal: str | None = None
    tone: str | None = None
    is_reply: bool = False
    draft_versions: list[DraftVersion] = Field(default_factory=list)
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    x_message_id: str | None = None
    created_at: datetime
    sent_at: datetime | None = None


class DenyRequest(ApiModel):
    feedback: str = Field(
        min_length=1, description="Why the draft was rejected, used to improve the next draft."
    )


class DispatchMode(ApiModel):
    dispatch: str = Field(
        default="blocking",
        pattern="^(blocking|queued)$",
        description=(
            "'blocking' sends within the request (respecting the pacing delay); "
            "'queued' hands off to the background drip dispatcher."
        ),
    )


class SendResult(ApiModel):
    id: uuid.UUID
    x_user_id: str
    status: str
    x_message_id: str | None = None
    error: str | None = None
    sent_at: datetime | None = None
    delay_applied_seconds: float = 0.0


class PollEventsResponse(ApiModel):
    polled: bool = False
    new_events: int = 0
    conversations_updated: int = 0
    drafts_created: int = 0
    next_poll_after_seconds: int = 0
    errors: list[str] = Field(default_factory=list)


class OutreachRecordOut(ApiModel):
    id: uuid.UUID
    x_user_id: str
    handle: str | None = None
    status: str
    goal: str | None = None
    tone: str | None = None
    is_reply: bool = False
    x_message_id: str | None = None
    error: str | None = None
    feedback: list[str] = Field(default_factory=list)
    created_at: datetime
    sent_at: datetime | None = None


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------
class SuppressionRequest(ApiModel):
    x_user_id: str | None = None
    handle: str | None = None
    reason: str = ""

    @field_validator("x_user_id", "handle")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SuppressionRecord(ApiModel):
    x_user_id: str
    handle: str | None = None
    reason: str | None = None
    created_at: datetime
