"""DM conversation and event state.

``GET /2/dm_events`` is limited to 15 requests per 15 minutes, so replies are
polled conservatively and every event is cached locally. New inbound messages
automatically produce a reply draft, which keeps the human in the loop even
for conversations the agent started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DmConversation, DmEvent
from app.services import pacing
from app.x.client import XClient
from app.x.errors import XApiError

log = structlog.get_logger(__name__)


@dataclass
class PollOutcome:
    """The result of one DM event poll."""

    polled: bool = False
    new_events: int = 0
    conversations_updated: int = 0
    inbound_messages: list[InboundMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class InboundMessage:
    """A new message from a recipient."""

    x_user_id: str
    event_id: str
    text: str
    sender_id: str
    created_at: datetime | None = None


async def poll_events(
    session: AsyncSession,
    client: XClient,
    *,
    max_results: int = 100,
) -> PollOutcome:
    """Poll the DM event stream and cache any new events."""
    outcome = PollOutcome()

    try:
        data = await client.poll_dm_events(max_results=max_results)
    except XApiError as exc:
        log.warning("dm_poll_failed", **exc.log_fields())
        outcome.errors.append(str(exc))
        return outcome

    outcome.polled = True
    await pacing.track_dm_events_polled(session)

    events = list(data.get("data") or [])
    conversations = data.get("conversations") or {}

    # Map conversation ids to the recipients they belong to.
    conversation_to_user: dict[str, str] = {}
    for conversation in conversations:
        conv_id = str(conversation.get("id") or "")
        if not conv_id:
            continue
        for participant in conversation.get("participant_ids") or []:
            conversation_to_user.setdefault(conv_id, str(participant))

    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue

        exists = await session.get(DmEvent, event_id)
        if exists is not None:
            continue

        event_type = str(event.get("event_type") or "MessageCreate")
        conversation_id: str | None = str(event.get("conversation_id") or "") or None
        sender_id: str | None = str(event.get("sender_id") or "") or None
        text = event.get("text")

        recipient_id = conversation_to_user.get(conversation_id or "") or ""
        if not recipient_id and conversation_id:
            existing = await session.execute(
                select(DmConversation).where(
                    DmConversation.conversation_id == conversation_id
                )
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                recipient_id = row.x_user_id

        session.add(
            DmEvent(
                id=event_id,
                conversation_id=conversation_id,
                x_user_id=recipient_id or None,
                sender_id=sender_id,
                text=text,
                event_type=event_type,
                created_at=_parse_datetime(event.get("created_at")),
            )
        )
        outcome.new_events += 1

        if event_type == "MessageCreate" and text and recipient_id:
            conversation_row = await _get_or_create_conversation(
                session, recipient_id, conversation_id
            )
            conversation_row.unread_inbound = True
            conversation_row.last_event_id = event_id
            conversation_row.last_polled_at = datetime.now(UTC)
            outcome.conversations_updated += 1
            outcome.inbound_messages.append(
                InboundMessage(
                    x_user_id=recipient_id,
                    event_id=event_id,
                    text=str(text),
                    sender_id=sender_id or "",
                    created_at=_parse_datetime(event.get("created_at")),
                )
            )

    await session.flush()
    log.info(
        "dm_poll_complete",
        new_events=outcome.new_events,
        conversations_updated=outcome.conversations_updated,
    )
    return outcome


async def _get_or_create_conversation(
    session: AsyncSession, x_user_id: str, conversation_id: str | None
) -> DmConversation:
    result = await session.execute(
        select(DmConversation).where(DmConversation.x_user_id == x_user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = DmConversation(x_user_id=x_user_id)
        session.add(row)
    if conversation_id and not row.conversation_id:
        row.conversation_id = conversation_id
    await session.flush()
    return row


async def conversation_history(
    session: AsyncSession, x_user_id: str
) -> list[dict[str, Any]]:
    """The cached message history for one recipient, oldest first."""
    result = await session.execute(
        select(DmEvent)
        .where(
            DmEvent.x_user_id == x_user_id,
            DmEvent.event_type == "MessageCreate",
        )
        .order_by(DmEvent.created_at.asc())
    )
    return [
        {
            "id": event.id,
            "sender_id": event.sender_id,
            "text": event.text,
            "created_at": event.created_at,
        }
        for event in result.scalars().all()
    ]


async def mark_conversation_read(session: AsyncSession, x_user_id: str) -> None:
    """Clear the unread flag after a reply draft has been created."""
    result = await session.execute(
        select(DmConversation).where(DmConversation.x_user_id == x_user_id)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        row.unread_inbound = False
        await session.flush()


async def conversations_needing_reply(
    session: AsyncSession,
) -> list[DmConversation]:
    """Conversations with an unanswered inbound message."""
    result = await session.execute(
        select(DmConversation).where(DmConversation.unread_inbound.is_(True))
    )
    return list(result.scalars().all())


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
