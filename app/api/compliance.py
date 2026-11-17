"""Compliance endpoints: the opt-out suppression list and the outreach ledger."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import NotFoundError, ValidationError
from app.models import OutreachRecord
from app.schemas import (
    OutreachRecordOut,
    SuppressionRecord,
    SuppressionRequest,
)
from app.services import suppression as suppression_service

router = APIRouter(prefix="/api/v1", tags=["compliance"])


@router.post(
    "/suppressions",
    response_model=SuppressionRecord,
    status_code=201,
    summary="Permanently suppress an account (Do Not Contact)",
)
async def add_suppression(
    request: SuppressionRequest,
    session: AsyncSession = Depends(get_session),
) -> SuppressionRecord:
    """Add an account to the permanent opt-out list.

    Suppressed accounts are excluded from outreach entirely, any pending draft
    is cancelled, and they are never contacted again.
    """
    if not request.x_user_id and not request.handle:
        raise ValidationError("Either x_user_id or handle must be provided")

    try:
        entry = await suppression_service.suppress(
            session,
            x_user_id=request.x_user_id,
            handle=request.handle,
            reason=request.reason,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    return SuppressionRecord(
        x_user_id=entry.x_user_id,
        handle=entry.handle,
        reason=entry.reason,
        created_at=entry.created_at,
    )


@router.get(
    "/suppressions",
    response_model=list[SuppressionRecord],
    summary="List suppressed accounts",
)
async def list_suppressions(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[SuppressionRecord]:
    entries = await suppression_service.list_suppressions(
        session, limit=limit, offset=offset
    )
    return [
        SuppressionRecord(
            x_user_id=entry.x_user_id,
            handle=entry.handle,
            reason=entry.reason,
            created_at=entry.created_at,
        )
        for entry in entries
    ]


@router.get(
    "/outreach/records",
    response_model=list[OutreachRecordOut],
    summary="The outreach ledger for one account, or all records",
)
async def list_records(
    x_user_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[OutreachRecordOut]:
    query = select(OutreachRecord)
    if x_user_id:
        query = query.where(OutreachRecord.x_user_id == x_user_id)
    if status:
        query = query.where(OutreachRecord.status == status)

    result = await session.execute(
        query.order_by(OutreachRecord.created_at.desc()).limit(limit).offset(offset)
    )
    return [
        OutreachRecordOut(
            id=record.id,
            x_user_id=record.x_user_id,
            handle=record.handle,
            status=record.status,
            goal=record.goal,
            tone=record.tone,
            is_reply=record.is_reply,
            x_message_id=record.x_message_id,
            error=record.error,
            feedback=record.feedback or [],
            created_at=record.created_at,
            sent_at=record.sent_at,
        )
        for record in result.scalars().all()
    ]


@router.get(
    "/outreach/records/{record_id}",
    response_model=OutreachRecordOut,
    summary="Get one outreach record",
)
async def get_record(
    record_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> OutreachRecordOut:
    record = await session.get(OutreachRecord, record_id)
    if record is None:
        raise NotFoundError(f"No outreach record with id {record_id}")
    return OutreachRecordOut(
        id=record.id,
        x_user_id=record.x_user_id,
        handle=record.handle,
        status=record.status,
        goal=record.goal,
        tone=record.tone,
        is_reply=record.is_reply,
        x_message_id=record.x_message_id,
        error=record.error,
        feedback=record.feedback or [],
        created_at=record.created_at,
        sent_at=record.sent_at,
    )
