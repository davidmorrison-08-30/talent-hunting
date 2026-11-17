"""PRIMARY ENDPOINT II: drafting, approving and sending direct messages.

The flow is deliberately human-in-the-loop, as required by
``build-docs/x-rules-and-limits.md``:

1. :func:`create_draft` writes a grounded draft and parks it in
   ``PENDING_APPROVAL``. Nothing is sent.
2. :func:`approve_and_send` re-checks every compliance guard, then sends the
   message with a paced delay.
3. :func:`deny_and_revise` takes the reviewer's feedback and produces a new
   draft for another round of approval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.profile import Evidence, KeyInformation, render_evidence, render_key_information
from app.config import Settings, get_settings
from app.errors import (
    DailyCapError,
    DuplicateOutreachError,
    NotFoundError,
    SuppressedError,
    ValidationError,
)
from app.llm.bedrock import BedrockLLM, get_llm
from app.llm.prompts import (
    ANTI_FABRICATION,
    DM_DRAFT_SCHEMA,
    DM_DRAFT_SYSTEM,
    DM_DRAFT_USER,
    DM_REVISION_SYSTEM,
    DM_REVISION_USER,
)
from app.models import (
    DiscoveredAccount,
    DmConversation,
    DmEvent,
    OutreachRecord,
    OutreachStatus,
)
from app.services import dedup, pacing
from app.x.client import XClient
from app.x.errors import BlockedDMError, XApiError

log = structlog.get_logger(__name__)


@dataclass
class DraftResult:
    """A generated draft plus the evidence a reviewer should check."""

    text: str
    citations: list[Evidence] = field(default_factory=list)
    personalization_notes: str = ""
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    conversation: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SendOutcome:
    """The result of an approval-and-send attempt."""

    record_id: uuid.UUID
    x_user_id: str
    status: str
    x_message_id: str | None = None
    error: str | None = None
    delay_applied_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------
async def gather_evidence(
    session: AsyncSession,
    client: XClient,
    x_user_id: str,
    *,
    max_posts: int = 10,
) -> tuple[KeyInformation, list[Evidence], list[dict[str, Any]]]:
    """Collect the key information, cited evidence and chat history.

    The stored key information from discovery is the primary source. Recent posts
    and the DM history are fetched fresh so the message reflects what the person
    is doing now.
    """
    result = await session.execute(
        select(DiscoveredAccount).where(DiscoveredAccount.x_user_id == x_user_id)
    )
    account = result.scalar_one_or_none()

    key_info = KeyInformation.from_payload(account.key_info if account else {})
    evidence = key_info.evidence

    # Fresh posts give the drafter something concrete and current to cite.
    try:
        tweets = await client.get_user_tweets(x_user_id, max_results=max_posts)
        for post in tweets.get("data") or []:
            text = (post.get("text") or "").strip()
            if not text:
                continue
            post_id = str(post.get("id") or "")
            evidence.append(
                Evidence(
                    post_id=post_id,
                    url=f"https://x.com/i/status/{post_id}" if post_id else "",
                    quote=text[:280],
                    supports="recent post",
                )
            )
    except XApiError as exc:
        log.warning("evidence_posts_failed", x_user_id=x_user_id, **exc.log_fields())

    conversation = await load_conversation(session, client, x_user_id)
    return key_info, evidence, conversation


async def load_conversation(
    session: AsyncSession, client: XClient, x_user_id: str
) -> list[dict[str, Any]]:
    """Load the cached DM history for a recipient, refreshing it when possible."""
    result = await session.execute(
        select(DmConversation).where(DmConversation.x_user_id == x_user_id)
    )
    conversation_row = result.scalar_one_or_none()

    conversation_id = conversation_row.conversation_id if conversation_row else None

    if conversation_id:
        try:
            data = await client.get_dm_events_for_conversation(conversation_id)
            events = list(data.get("data") or [])
            if events:
                assert conversation_row is not None  # noqa: S101 - implied by the id
                await _store_events(session, conversation_row, events)
                conversation = _events_to_messages(events)
                if conversation:
                    return conversation
        except XApiError as exc:
            log.warning(
                "conversation_refresh_failed", x_user_id=x_user_id, **exc.log_fields()
            )

    # No conversation id yet: try the participant-scoped endpoint.
    try:
        data = await client.get_dm_events_for_participant(x_user_id)
        events = list(data.get("data") or [])
        if events:
            if conversation_row is None:
                conversation_row = DmConversation(x_user_id=x_user_id)
                session.add(conversation_row)
            conversation_id = (data.get("meta") or {}).get("conversation_id") or (
                conversation_id
            )
            await _store_events(session, conversation_row, events)
            return _events_to_messages(events)
    except XApiError as exc:
        log.warning("participant_events_failed", x_user_id=x_user_id, **exc.log_fields())

    # Fall back to whatever is cached locally.
    cached = await session.execute(
        select(DmEvent)
        .where(DmEvent.x_user_id == x_user_id)
        .order_by(DmEvent.created_at.asc())
    )
    return _events_to_messages(list(cached.scalars().all()))


async def _store_events(
    session: AsyncSession,
    conversation_row: DmConversation,
    events: list[dict[str, Any]],
) -> None:
    """Persist DM events into the local cache."""
    if conversation_row is None:
        return
    conversation_row.last_polled_at = datetime.now(UTC)

    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        exists = await session.get(DmEvent, event_id)
        if exists is not None:
            continue
        session.add(
            DmEvent(
                id=event_id,
                conversation_id=conversation_row.conversation_id,
                x_user_id=conversation_row.x_user_id,
                sender_id=str(event.get("sender_id") or "") or None,
                text=event.get("text"),
                event_type=str(event.get("event_type") or "MessageCreate"),
                created_at=_parse_datetime(event.get("created_at")),
            )
        )
    await session.flush()


def _events_to_messages(events: list[Any]) -> list[dict[str, Any]]:
    """Convert DM events into a simple message list for the prompt."""
    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = (
            event.get("event_type")
            if isinstance(event, dict)
            else getattr(event, "event_type", None)
        )
        if event_type != "MessageCreate":
            continue
        if isinstance(event, dict):
            text = event.get("text") or ""
            sender = event.get("sender_id")
        else:
            text = getattr(event, "text", "") or ""
            sender = getattr(event, "sender_id", None)
        messages.append({"sender_id": sender, "text": text})
    return messages


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


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------
async def create_draft(
    session: AsyncSession,
    client: XClient,
    *,
    x_user_id: str,
    goal: str = "",
    tone: str = "concise and warm",
    is_reply: bool = False,
    llm: BedrockLLM | None = None,
) -> OutreachRecord:
    """Create a draft and park it in ``PENDING_APPROVAL``.

    Pre-flight checks reject suppressed accounts, accounts that already have a
    live outreach record, and days where the DM cap has been reached.
    """
    llm = llm or get_llm()

    await dedup.assert_can_contact(session, x_user_id)
    await pacing.assert_daily_cap_available(session)

    key_info, evidence, conversation = await gather_evidence(session, client, x_user_id)

    result = await llm.converse_json(
        system=DM_DRAFT_SYSTEM,
        user=DM_DRAFT_USER.format(
            goal_block=f"Goal of the message: {goal}\n" if goal else "",
            tone_block=f"Tone: {tone}\n" if tone else "",
            evidence=render_evidence(evidence),
            key_information=render_key_information(key_info),
            conversation=(
                "\n".join(f"- {m.get('text', '')}" for m in conversation[-10:])
                if conversation
                else "(no prior conversation)"
            ),
            anti_fabrication=ANTI_FABRICATION,
        ),
        schema=DM_DRAFT_SCHEMA,
    )

    text = str(result.get("message") or "").strip()
    if not text:
        raise ValidationError("The model did not produce a message body")

    citations = [
        Evidence(
            post_id=str(c.get("post_id") or ""),
            url=str(c.get("url") or ""),
            quote=str(c.get("quote") or ""),
        )
        for c in (result.get("citations") or [])
        if isinstance(c, dict)
    ]

    account_result = await session.execute(
        select(DiscoveredAccount).where(DiscoveredAccount.x_user_id == x_user_id)
    )
    account = account_result.scalar_one_or_none()

    record = await dedup.create_record(
        session,
        x_user_id=x_user_id,
        handle=account.username if account else None,
        goal=goal or None,
        tone=tone or None,
        status=OutreachStatus.PENDING_APPROVAL,
        is_reply=is_reply,
    )

    record.draft_versions = [
        {
            "version": 1,
            "text": text,
            "citations": [c.to_payload() for c in citations],
            "personalization_notes": str(result.get("personalization_notes") or ""),
            "feedback": "",
            "created_at": datetime.now(UTC).isoformat(),
        }
    ]
    record.evidence_snapshot = {
        "key_information": key_info.to_payload(),
        "evidence": [e.to_payload() for e in evidence],
        "conversation": conversation,
    }
    await session.flush()

    log.info("draft_created", record_id=str(record.id), x_user_id=x_user_id)
    return record


async def revise_draft(
    session: AsyncSession,
    *,
    record: OutreachRecord,
    feedback: str,
    llm: BedrockLLM | None = None,
) -> OutreachRecord:
    """Regenerate a draft using the reviewer's feedback."""
    llm = llm or get_llm()

    versions = list(record.draft_versions or [])
    if not versions:
        raise NotFoundError("This record has no draft to revise")

    original = versions[-1]
    snapshot = record.evidence_snapshot or {}
    key_info = KeyInformation.from_payload(snapshot.get("key_information") or {})
    evidence = [
        Evidence(
            post_id=str(e.get("post_id") or ""),
            url=str(e.get("url") or ""),
            quote=str(e.get("quote") or ""),
        )
        for e in (snapshot.get("evidence") or [])
        if isinstance(e, dict)
    ]

    result = await llm.converse_json(
        system=DM_REVISION_SYSTEM,
        user=DM_REVISION_USER.format(
            original_message=original.get("text", ""),
            feedback=feedback,
            evidence=render_evidence(evidence),
            key_information=render_key_information(key_info),
            anti_fabrication=ANTI_FABRICATION,
        ),
        schema=DM_DRAFT_SCHEMA,
    )

    text = str(result.get("message") or "").strip()
    if not text:
        raise ValidationError("The model did not produce a revised message body")

    citations = [
        {
            "post_id": str(c.get("post_id") or ""),
            "url": str(c.get("url") or ""),
            "quote": str(c.get("quote") or ""),
        }
        for c in (result.get("citations") or [])
        if isinstance(c, dict)
    ]

    versions.append(
        {
            "version": len(versions) + 1,
            "text": text,
            "citations": citations,
            "personalization_notes": str(result.get("personalization_notes") or ""),
            "feedback": feedback,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    record.draft_versions = versions
    record.feedback = list(record.feedback or []) + [feedback]
    record.status = OutreachStatus.PENDING_APPROVAL.value
    record.error = None
    await session.flush()

    log.info(
        "draft_revised",
        record_id=str(record.id),
        version=len(versions),
    )
    return record


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
async def approve_and_send(
    session: AsyncSession,
    client: XClient,
    *,
    record: OutreachRecord,
    settings: Settings | None = None,
    apply_pacing_delay: bool = True,
) -> SendOutcome:
    """Send an approved draft, enforcing every compliance guard.

    The order of checks matters: the dedup and suppression checks are repeated
    immediately before sending so a change between drafting and approval cannot
    slip through.
    """
    settings = settings or get_settings()

    if record.status == OutreachStatus.SENT.value:
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            x_message_id=record.x_message_id,
            error="This message was already sent",
        )

    if record.status not in (
        OutreachStatus.PENDING_APPROVAL.value,
        OutreachStatus.APPROVED.value,
    ):
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=f"This record is in state {record.status} and cannot be sent",
        )

    # Re-check the guards right before sending. The record under review is
    # excluded from the dedup lookup so it does not block its own send.
    try:
        await dedup.assert_can_contact(session, record.x_user_id, exclude_id=record.id)
    except SuppressedError:
        record.status = OutreachStatus.OPTED_OUT.value
        record.error = "This account opted out and must not be contacted"
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=record.error,
        )
    except DuplicateOutreachError as exc:
        record.status = OutreachStatus.FAILED.value
        record.error = str(exc)
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=str(exc),
        )

    try:
        await pacing.assert_daily_cap_available(session)
    except DailyCapError as exc:
        record.status = OutreachStatus.FAILED.value
        record.error = str(exc)
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=str(exc),
        )

    versions = list(record.draft_versions or [])
    if not versions:
        record.status = OutreachStatus.FAILED.value
        record.error = "There is no message body to send"
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=record.error,
        )

    text = str(versions[-1].get("text") or "").strip()
    if not text:
        record.status = OutreachStatus.FAILED.value
        record.error = "The message body is empty"
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=record.error,
        )

    # Anti-copypasta check: reject a draft that repeats a recent send.
    duplicate = await find_near_duplicate(session, text, exclude_id=record.id)
    if duplicate is not None:
        record.status = OutreachStatus.PENDING_APPROVAL.value
        record.error = (
            "This message is too similar to one already sent to "
            f"@{duplicate.handle or duplicate.x_user_id}. Please revise it."
        )
        await session.flush()
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=record.error,
        )

    delay = 0.0
    if apply_pacing_delay:
        delay = await pacing.wait_before_send(settings)

    try:
        response = await client.send_dm(record.x_user_id, text)
    except BlockedDMError as exc:
        record.status = OutreachStatus.BLOCKED.value
        record.error = str(exc)
        await session.flush()
        log.info("dm_blocked", x_user_id=record.x_user_id)
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=str(exc),
            delay_applied_seconds=delay,
        )
    except XApiError as exc:
        record.status = OutreachStatus.FAILED.value
        record.error = str(exc)
        await session.flush()
        log.warning("dm_send_failed", x_user_id=record.x_user_id, **exc.log_fields())
        return SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=record.status,
            error=str(exc),
            delay_applied_seconds=delay,
        )

    event = (response.get("data") or {}) if isinstance(response, dict) else {}
    record.status = OutreachStatus.SENT.value
    record.x_message_id = str(event.get("id") or "") or None
    record.sent_at = datetime.now(UTC)
    record.error = None
    await session.flush()

    await pacing.record_dm_sent(session)
    log.info(
        "dm_sent",
        record_id=str(record.id),
        x_user_id=record.x_user_id,
        delay=round(delay, 1),
    )
    return SendOutcome(
        record_id=record.id,
        x_user_id=record.x_user_id,
        status=record.status,
        x_message_id=record.x_message_id,
        delay_applied_seconds=delay,
    )


async def find_near_duplicate(
    session: AsyncSession, text: str, *, exclude_id: uuid.UUID | None = None
) -> OutreachRecord | None:
    """Find a recently sent message that is near-identical to ``text``."""
    settings = get_settings()
    result = await session.execute(
        select(OutreachRecord)
        .where(OutreachRecord.status == OutreachStatus.SENT.value)
        .order_by(OutreachRecord.sent_at.desc())
        .limit(50)
    )

    normalized = " ".join(text.lower().split())
    for record in result.scalars().all():
        if exclude_id is not None and record.id == exclude_id:
            continue
        versions = list(record.draft_versions or [])
        if not versions:
            continue
        sent_text = str(versions[-1].get("text") or "")
        ratio = SequenceMatcher(None, normalized, " ".join(sent_text.lower().split())).ratio()
        if ratio >= settings.dm_similarity_threshold:
            return record
    return None
