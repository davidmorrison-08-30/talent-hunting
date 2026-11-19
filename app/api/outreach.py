"""PRIMARY ENDPOINT II: human-in-the-loop outreach endpoints."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import drafts as draft_agent
from app.config import Settings, get_settings
from app.db import get_session
from app.errors import ConflictError, NotFoundError, ValidationError
from app.models import OutreachRecord, OutreachStatus
from app.schemas import (
    DenyRequest,
    DraftRequest,
    DraftResponse,
    DraftVersion,
    EvidenceItem,
    PollEventsResponse,
    SendResult,
)
from app.x.client import XClient
from app.x.errors import x_error_fields

router = APIRouter(prefix="/api/v1/outreach", tags=["outreach"])

log = structlog.get_logger(__name__)

#: The statuses of a record that is still waiting for a human decision. Such a
#: record is the one draft an account has, and it is the only kind that can be
#: discarded without touching the audit trail of what was actually sent.
_UNAPPROVED_STATUSES = (
    OutreachStatus.DRAFT.value,
    OutreachStatus.PENDING_APPROVAL.value,
)


def _to_response(record: OutreachRecord) -> DraftResponse:
    """Map an outreach record onto the public response model."""
    versions = [
        DraftVersion(
            version=int(v.get("version") or index + 1),
            text=str(v.get("text") or ""),
            citations=[EvidenceItem(**c) for c in (v.get("citations") or [])],
            personalization_notes=str(v.get("personalization_notes") or ""),
            feedback=str(v.get("feedback") or ""),
            created_at=v.get("created_at"),
        )
        for index, v in enumerate(record.draft_versions or [])
    ]
    snapshot = record.evidence_snapshot or {}
    return DraftResponse(
        id=record.id,
        x_user_id=record.x_user_id,
        handle=record.handle,
        status=record.status,
        goal=record.goal,
        tone=record.tone,
        is_reply=record.is_reply,
        draft_versions=versions,
        evidence_snapshot=snapshot,
        conversation=snapshot.get("conversation") or [],
        error=record.error,
        x_message_id=record.x_message_id,
        created_at=record.created_at,
        sent_at=record.sent_at,
    )


@router.post(
    "/drafts",
    response_model=DraftResponse,
    status_code=201,
    summary="Draft a personalized direct message for human approval",
)
async def create_draft(
    request: DraftRequest,
    session: AsyncSession = Depends(get_session),
) -> DraftResponse:
    """Write a grounded draft and park it in ``PENDING_APPROVAL``.

    Nothing is sent by this call. The draft references real evidence from the
    recipient's profile and recent posts, and it includes the evidence so a
    reviewer can verify the personalization before approving it.
    """
    if not request.x_user_id.strip():
        raise ValidationError("x_user_id is required")

    async with XClient(session) as client:
        record = await draft_agent.create_draft(
            session,
            client,
            x_user_id=request.x_user_id.strip(),
            goal=request.goal,
            tone=request.tone,
            is_reply=request.is_reply,
        )
    return _to_response(record)


@router.get(
    "/drafts",
    response_model=list[DraftResponse],
    summary="List outreach records, filtered by status",
)
async def list_drafts(
    status: str | None = Query(
        default=None,
        description="Filter by status, e.g. PENDING_APPROVAL, APPROVED or SENT.",
    ),
    x_user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[DraftResponse]:
    query = select(OutreachRecord)
    if status:
        query = query.where(OutreachRecord.status == status)
    if x_user_id:
        query = query.where(OutreachRecord.x_user_id == x_user_id)

    result = await session.execute(
        query.order_by(OutreachRecord.created_at.desc()).limit(limit).offset(offset)
    )
    return [_to_response(record) for record in result.scalars().all()]


@router.get(
    "/drafts/current",
    response_model=DraftResponse,
    summary="The one draft currently waiting for a decision on an account",
)
async def current_draft(
    x_user_id: str = Query(
        ..., description="The recipient whose pending draft should be returned."
    ),
    session: AsyncSession = Depends(get_session),
) -> DraftResponse:
    """Return the single unapproved draft for a recipient.

    There is no global approval queue any more: each account has at most one
    draft waiting for a decision, and the chat panel asks for it when an
    account is selected. The newest unapproved record wins, so a revision made
    with reviewer feedback is the one that is shown.
    """
    result = await session.execute(
        select(OutreachRecord)
        .where(
            OutreachRecord.x_user_id == x_user_id,
            OutreachRecord.status.in_(_UNAPPROVED_STATUSES),
        )
        .order_by(OutreachRecord.created_at.desc())
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise NotFoundError(f"No pending draft for {x_user_id}")
    return _to_response(record)


@router.get(
    "/drafts/{record_id}",
    response_model=DraftResponse,
    summary="Get one outreach record with its evidence",
)
async def get_draft(
    record_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DraftResponse:
    record = await session.get(OutreachRecord, record_id)
    if record is None:
        raise NotFoundError(f"No outreach record with id {record_id}")
    return _to_response(record)


@router.post(
    "/drafts/{record_id}/approve",
    response_model=SendResult,
    summary="Approve a draft and send it (paced, with all guards re-checked)",
)
async def approve_draft(
    record_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SendResult:
    """Approve and send a drafted message.

    The message goes out within the request, honouring the randomized 30-90
    second pacing delay. The dedup lock, the suppression list and the daily cap
    are re-checked immediately before the message goes out.
    """
    record = await session.get(OutreachRecord, record_id)
    if record is None:
        raise NotFoundError(f"No outreach record with id {record_id}")

    async with XClient(session) as client:
        outcome = await draft_agent.approve_and_send(
            session, client, record=record, settings=settings
        )

    return SendResult(
        id=outcome.record_id,
        x_user_id=outcome.x_user_id,
        status=outcome.status,
        x_message_id=outcome.x_message_id,
        error=outcome.error,
        sent_at=record.sent_at,
        delay_applied_seconds=outcome.delay_applied_seconds,
    )


@router.post(
    "/drafts/{record_id}/deny",
    response_model=DraftResponse,
    summary="Reject a draft with feedback and get an improved revision",
)
async def deny_draft(
    record_id: uuid.UUID,
    request: DenyRequest,
    session: AsyncSession = Depends(get_session),
) -> DraftResponse:
    """Reject a draft and regenerate it using the reviewer's feedback.

    The previous version is kept in the version history so the whole review
    trail stays auditable. The new version is again parked in
    ``PENDING_APPROVAL`` and is not sent.
    """
    record = await session.get(OutreachRecord, record_id)
    if record is None:
        raise NotFoundError(f"No outreach record with id {record_id}")

    if record.status == OutreachStatus.SENT.value:
        raise ConflictError(
            "This message was already sent and cannot be revised",
            details={"record_id": str(record.id)},
        )

    await draft_agent.revise_draft(session, record=record, feedback=request.feedback)
    return _to_response(record)


@router.post(
    "/poll-events",
    response_model=PollEventsResponse,
    summary="Poll for inbound replies and draft responses for human approval",
)
async def poll_events(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> PollEventsResponse:
    """Poll the DM event stream for new inbound messages.

    ``GET /2/dm_events`` is limited to 15 calls per 15 minutes, so the response
    reports when the next poll is due. Each new inbound message produces a
    reply draft in ``PENDING_APPROVAL``; nothing is sent automatically.
    """
    from app.services import conversations as conversation_service

    async with XClient(session) as client:
        outcome = await conversation_service.poll_events(session, client)

    drafts_created = 0
    for inbound in outcome.inbound_messages:
        try:
            await conversation_service.mark_conversation_read(
                session, inbound.x_user_id
            )
            await draft_agent.create_draft(
                session,
                client,
                x_user_id=inbound.x_user_id,
                goal="Reply to the inbound message.",
                tone="conversational and warm",
                is_reply=True,
            )
            drafts_created += 1
        except Exception as exc:  # noqa: BLE001
            # A draft can fail if the recipient is suppressed or already has a
            # live record; that must not abort the whole poll.
            log.warning("reply_draft_failed", x_user_id=inbound.x_user_id, **x_error_fields(exc))
            outcome.errors.append(str(exc))

    return PollEventsResponse(
        polled=outcome.polled,
        new_events=outcome.new_events,
        conversations_updated=outcome.conversations_updated,
        drafts_created=drafts_created,
        next_poll_after_seconds=settings.dm_poll_interval_seconds,
        errors=outcome.errors,
    )


@router.delete(
    "/drafts/{record_id}",
    status_code=204,
    summary="Discard an unapproved draft",
)
async def delete_draft(
    record_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Delete a draft that has not been approved yet.

    Switching to another account discards the draft that was still waiting for
    a decision, so the record is deleted rather than left behind as an
    invisible pending draft that would block the next one for the same
    recipient. A record that was already sent (or otherwise reached a terminal
    state) is never deleted: it is the audit trail of what went out.
    """
    record = await session.get(OutreachRecord, record_id)
    if record is None:
        raise NotFoundError(f"No outreach record with id {record_id}")

    if record.status not in _UNAPPROVED_STATUSES:
        raise ConflictError(
            "This message was already approved or sent and cannot be discarded",
            details={"record_id": str(record.id), "status": record.status},
        )

    await session.delete(record)
    await session.flush()
    log.info("draft_discarded", record_id=str(record.id), x_user_id=record.x_user_id)
    return Response(status_code=204)


@router.get("/budget", summary="Today's dispatch and read budget")
async def budget(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Show how much of today's dispatch and pay-per-use budget is used."""
    from app.services.pacing import get_budget

    budget_row = await get_budget(session)
    return {
        "day": budget_row.day,
        "dm_sent": budget_row.dm_sent_count,
        "dm_daily_cap": settings.dm_daily_cap,
        "dm_remaining": max(0, settings.dm_daily_cap - budget_row.dm_sent_count),
        "posts_read": budget_row.posts_read_count,
        "users_read": budget_row.users_read_count,
        "dm_events_polled": budget_row.dm_events_polled,
        "estimated_cost_usd": round(budget_row.estimated_cost_usd, 4),
        "pacing_delay_seconds": [
            settings.dm_min_delay_seconds,
            settings.dm_max_delay_seconds,
        ],
    }


@router.get(
    "/conversations/{x_user_id}",
    summary="The cached DM history with one recipient",
)
async def conversation(
    x_user_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    """Return the locally cached DM history for a recipient.

    Both directions are included: the messages the operator sent are cached
    when they are approved, and the inbound messages are cached by the DM
    event poll. Each row carries ``outbound`` so the caller can tell the two
    apart without knowing the operator's own id.
    """
    from app.services import conversations as conversation_service

    return await conversation_service.conversation_history(session, x_user_id)
