"""SQLAlchemy ORM models.

The schema encodes the compliance invariants from `build-docs/x-rules-and-limits.md`:

* a recipient may only ever have one live outreach record (partial unique index),
* opt-outs live in a separate permanent suppression table,
* daily DM volume and pay-per-use read volume are tracked in `rate_budget`.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OutreachStatus(enum.StrEnum):
    """Lifecycle of a DM outreach record."""

    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    SENT = "SENT"
    DENIED = "DENIED"
    BLOCKED = "BLOCKED"
    OPTED_OUT = "OPTED_OUT"
    FAILED = "FAILED"


#: Statuses that block any further outreach to the same recipient.
ACTIVE_OUTREACH_STATUSES = (
    OutreachStatus.PENDING_APPROVAL,
    OutreachStatus.SENT,
    OutreachStatus.OPTED_OUT,
)


class BotVerdict(enum.StrEnum):
    HUMAN = "human"
    LIKELY_BOT = "likely_bot"
    UNKNOWN = "unknown"


class OAuthToken(Base):
    """OAuth 2.0 user-context token pair.

    X refresh tokens are single-use, so each refresh overwrites the stored one.
    """

    __tablename__ = "oauth_tokens"

    x_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    x_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    scopes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )


class DiscoveryRun(Base):
    """Audit record for one discovery job."""

    __tablename__ = "discovery_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    input_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_segment: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    seed_accounts: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    accounts: Mapped[list[DiscoveredAccount]] = relationship(back_populates="run")


class DiscoveredAccount(Base):
    """An account found by the discovery agent, with its key information."""

    __tablename__ = "discovered_accounts"

    x_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    followers_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    following_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tweets_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at_x: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    public_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    bot_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bot_verdict: Mapped[str] = mapped_column(
        String(16), default=BotVerdict.UNKNOWN.value, nullable=False, index=True
    )
    bot_reasons: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    key_info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    segment_tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("discovery_runs.id", ondelete="SET NULL"), nullable=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    run: Mapped[DiscoveryRun | None] = relationship(back_populates="accounts")

    __table_args__ = (
        Index("ix_discovered_accounts_bot_followers", "bot_verdict", "followers_count"),
    )


class OutreachRecord(Base):
    """The DM outreach ledger: draft lifecycle, dedup, and send audit."""

    __tablename__ = "outreach_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    x_user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    handle: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(24), default=OutreachStatus.DRAFT.value, nullable=False, index=True
    )
    draft_versions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    evidence_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    tone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    x_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    is_reply: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=sa.false(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Hard dedup lock: at most one live record per recipient.
        Index(
            "uq_outreach_active_x_user_id",
            "x_user_id",
            unique=True,
                postgresql_where=(
                    "status IN ('PENDING_APPROVAL', 'SENT', 'OPTED_OUT')"
                ),
        ),
        Index("ix_outreach_status_created", "status", "created_at"),
    )


class SuppressionEntry(Base):
    """Permanent opt-out list. Entries are never removed automatically."""

    __tablename__ = "suppression_list"

    x_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    handle: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


class DmConversation(Base):
    """Cached DM conversation state for one recipient."""

    __tablename__ = "dm_conversations"

    x_user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    last_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    unread_inbound: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )


class DmEvent(Base):
    """Cached DM event log (messages and participant changes)."""

    __tablename__ = "dm_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    x_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    sender_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_dm_events_conversation_created", "conversation_id", "created_at"),
    )


class RateBudget(Base):
    """Persisted daily spend and dispatch counters."""

    __tablename__ = "rate_budget"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    dm_sent_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    posts_read_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    users_read_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dm_events_polled: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )


class OAuthState(Base):
    """Short-lived PKCE `state` values for the OAuth 2.0 authorization flow."""

    __tablename__ = "oauth_states"

    state: Mapped[str] = mapped_column(String(128), primary_key=True)
    code_verifier: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
