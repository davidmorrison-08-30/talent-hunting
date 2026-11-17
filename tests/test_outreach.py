"""Tests for the outreach flow: dedup, suppression, pacing, approval and denial."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
import respx

from app.agents import drafts as draft_agent
from app.models import OutreachRecord, OutreachStatus, SuppressionEntry
from app.services import dedup, pacing
from app.x.client import XClient


class FakeLLM:
    """A BedrockLLM stand-in with prompt-keyed responses."""

    def __init__(self, by_prompt: dict[str, dict[str, Any]]) -> None:
        self.by_prompt = by_prompt
        self.calls: list[dict[str, Any]] = []

    async def converse_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        user = str(kwargs.get("user", ""))
        system = str(kwargs.get("system", ""))
        for marker, response in self.by_prompt.items():
            if marker in user or marker in system:
                return response
        return {}


DRAFT_RESPONSE = {
    "message": (
        "Saw your post about the new IK solver - congrats on shipping that. "
        "We run a robotics meetup in Singapore and would love to have you speak. "
        "Are you up for a chat?"
    ),
    "citations": [
        {
            "post_id": "post-0",
            "url": "https://x.com/alice/status/post-0",
            "quote": "Shipped a new IK solver today",
        }
    ],
    "personalization_notes": "Referenced the IK solver post.",
}

REVISION_RESPONSE = {
    "message": "Your IK solver post caught my eye - impressive work. Interested in our meetup?",
    "citations": [
        {
            "post_id": "post-0",
            "url": "https://x.com/alice/status/post-0",
            "quote": "Shipped a new IK solver today",
        }
    ],
    "personalization_notes": "Shortened and made it less formal.",
}


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    llm = FakeLLM(
        {
            "Draft a direct message": DRAFT_RESPONSE,
            "Revise this rejected message": REVISION_RESPONSE,
        }
    )
    monkeypatch.setattr("app.agents.drafts.get_llm", lambda: llm)
    return llm


@pytest.fixture
def no_pacing_delay(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pacing.next_delay_seconds", lambda settings=None: 0.0)


@pytest.fixture
def x_api_mock(respx_mock: respx.MockRouter) -> respx.MockRouter:
    respx_mock.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "post-0",
                        "created_at": "2026-01-01T00:00:00.000Z",
                        "text": "Shipped a new IK solver today",
                        "public_metrics": {},
                    }
                ]
            },
        )
    )
    respx_mock.get(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/dm_events"
    ).mock(return_value=httpx.Response(200, json={"data": []}))
    respx_mock.get(
        url__regex=r"https://api\.x\.com/2/dm_conversations/[^/]+/dm_events"
    ).mock(return_value=httpx.Response(200, json={"data": []}))
    return respx_mock


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_assert_can_contact_rejects_duplicate(db_session) -> None:
    await dedup.create_record(
        db_session, x_user_id="u1", status=OutreachStatus.PENDING_APPROVAL
    )
    await db_session.flush()

    with pytest.raises(dedup.DuplicateOutreachError):
        await dedup.assert_can_contact(db_session, "u1")


@pytest.mark.asyncio
async def test_assert_can_contact_allows_new_recipient(db_session) -> None:
    await dedup.assert_can_contact(db_session, "u1")


@pytest.mark.asyncio
async def test_denied_record_does_not_block_a_new_draft(db_session) -> None:
    record = await dedup.create_record(
        db_session, x_user_id="u1", status=OutreachStatus.DRAFT
    )
    record.status = OutreachStatus.DENIED.value
    await db_session.flush()

    # A DENIED record is not live, so a fresh draft is allowed.
    await dedup.assert_can_contact(db_session, "u1")


@pytest.mark.asyncio
async def test_assert_can_contact_rejects_suppressed(db_session) -> None:
    db_session.add(SuppressionEntry(x_user_id="u1", handle="alice", reason="opt out"))
    await db_session.flush()

    with pytest.raises(dedup.SuppressedError):
        await dedup.assert_can_contact(db_session, "u1")


@pytest.mark.asyncio
async def test_database_index_blocks_a_second_live_record(db_session) -> None:
    from sqlalchemy.exc import IntegrityError

    await dedup.create_record(
        db_session, x_user_id="u1", status=OutreachStatus.PENDING_APPROVAL
    )
    await db_session.flush()

    db_session.add(
        OutreachRecord(
            id=uuid.uuid4(),
            x_user_id="u1",
            status=OutreachStatus.SENT.value,
            draft_versions=[],
            evidence_snapshot={},
            feedback=[],
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_suppress_moves_pending_draft_to_opted_out(db_session) -> None:
    from app.services import suppression

    await dedup.create_record(
        db_session, x_user_id="u1", status=OutreachStatus.PENDING_APPROVAL
    )
    await db_session.flush()

    await suppression.suppress(db_session, x_user_id="u1", reason="asked to stop")
    await db_session.flush()

    record = await db_session.get(OutreachRecord, await _first_record_id(db_session))
    assert record is not None
    assert record.status == OutreachStatus.OPTED_OUT.value


async def _first_record_id(session) -> uuid.UUID:
    from sqlalchemy import select

    result = await session.execute(select(OutreachRecord))
    return result.scalars().first().id


# ---------------------------------------------------------------------------
# Pacing / daily cap
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_daily_cap_blocks_sending(db_session, monkeypatch) -> None:
    from app.config import get_settings

    settings = get_settings()
    budget = await pacing.get_budget(db_session)
    budget.dm_sent_count = settings.dm_daily_cap
    await db_session.flush()

    with pytest.raises(pacing.DailyCapError):
        await pacing.assert_daily_cap_available(db_session)


@pytest.mark.asyncio
async def test_daily_cap_allows_below_the_limit(db_session) -> None:
    await pacing.assert_daily_cap_available(db_session)


@pytest.mark.asyncio
async def test_record_dm_sent_increments_the_counter(db_session) -> None:
    before = await pacing.dm_sent_today(db_session)
    after = await pacing.record_dm_sent(db_session)
    assert after == before + 1


def test_next_delay_is_within_the_configured_window() -> None:
    from app.config import get_settings

    settings = get_settings()
    for _ in range(20):
        delay = pacing.next_delay_seconds(settings)
        assert settings.dm_min_delay_seconds <= delay <= settings.dm_max_delay_seconds


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_create_draft_parks_it_for_approval(
    db_session, auth_token, fake_llm, x_api_mock
) -> None:
    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(
            db_session,
            client,
            x_user_id="u1",
            goal="Invite them to a robotics meetup",
        )

    assert record.status == OutreachStatus.PENDING_APPROVAL.value
    assert len(record.draft_versions) == 1
    assert "IK solver" in record.draft_versions[0]["text"]
    assert record.draft_versions[0]["citations"]
    assert record.evidence_snapshot["key_information"]
    assert record.evidence_snapshot["evidence"]


@pytest.mark.asyncio
@respx.mock
async def test_create_draft_refuses_a_duplicate(
    db_session, auth_token, fake_llm, x_api_mock
) -> None:
    async with XClient(db_session) as client:
        await draft_agent.create_draft(db_session, client, x_user_id="u1")
        with pytest.raises(dedup.DuplicateOutreachError):
            await draft_agent.create_draft(db_session, client, x_user_id="u1")


@pytest.mark.asyncio
@respx.mock
async def test_create_draft_refuses_a_suppressed_recipient(
    db_session, auth_token, fake_llm, x_api_mock
) -> None:
    db_session.add(SuppressionEntry(x_user_id="u1", handle="alice"))
    await db_session.flush()

    async with XClient(db_session) as client:
        with pytest.raises(dedup.SuppressedError):
            await draft_agent.create_draft(db_session, client, x_user_id="u1")


# ---------------------------------------------------------------------------
# Denial and revision
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_revise_draft_keeps_history_and_asks_again(
    db_session, auth_token, fake_llm, x_api_mock
) -> None:
    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")
        await draft_agent.revise_draft(
            db_session, record=record, feedback="too formal, cut the second paragraph"
        )

    assert len(record.draft_versions) == 2
    assert record.draft_versions[0]["text"] != record.draft_versions[1]["text"]
    assert record.draft_versions[1]["feedback"] == "too formal, cut the second paragraph"
    assert record.status == OutreachStatus.PENDING_APPROVAL.value
    assert record.feedback == ["too formal, cut the second paragraph"]


# ---------------------------------------------------------------------------
# Approval and sending
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_approve_sends_the_message(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    send = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")
        outcome = await draft_agent.approve_and_send(
            db_session, client, record=record
        )

    assert outcome.status == OutreachStatus.SENT.value
    assert outcome.x_message_id == "msg-1"
    assert record.sent_at is not None
    assert send.called
    assert await pacing.dm_sent_today(db_session) == 1


@pytest.mark.asyncio
@respx.mock
async def test_approve_marks_blocked_dm(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(
        return_value=httpx.Response(
            403, json={"errors": [{"title": "Forbidden", "detail": "cannot message user"}]}
        )
    )

    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")
        outcome = await draft_agent.approve_and_send(
            db_session, client, record=record
        )

    assert outcome.status == OutreachStatus.BLOCKED.value
    assert "does not accept direct messages" in (outcome.error or "")


@pytest.mark.asyncio
@respx.mock
async def test_approve_refuses_when_suppressed_between_draft_and_send(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    send = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")

        # The recipient opts out after the draft was written.
        db_session.add(SuppressionEntry(x_user_id="u1", handle="alice"))
        await db_session.flush()

        outcome = await draft_agent.approve_and_send(
            db_session, client, record=record
        )

    assert outcome.status != OutreachStatus.SENT.value
    assert not send.called


@pytest.mark.asyncio
@respx.mock
async def test_approve_rejects_a_near_duplicate_of_a_previous_send(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    send = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        first = await draft_agent.create_draft(db_session, client, x_user_id="u1")
        await draft_agent.approve_and_send(db_session, client, record=first)

        second = await draft_agent.create_draft(db_session, client, x_user_id="u2")
        outcome = await draft_agent.approve_and_send(
            db_session, client, record=second
        )

    assert outcome.status == OutreachStatus.PENDING_APPROVAL.value
    assert "too similar" in (outcome.error or "")
    assert send.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_approve_does_not_resend_an_already_sent_message(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    send = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")
        await draft_agent.approve_and_send(db_session, client, record=record)
        outcome = await draft_agent.approve_and_send(
            db_session, client, record=record
        )

    assert outcome.status == OutreachStatus.SENT.value
    assert "already sent" in (outcome.error or "")
    assert send.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_approve_respects_the_daily_cap(
    db_session, auth_token, fake_llm, x_api_mock, no_pacing_delay
) -> None:
    from app.config import get_settings

    send = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        # The draft is written while there is still budget.
        record = await draft_agent.create_draft(db_session, client, x_user_id="u1")

        # The daily budget is exhausted between drafting and approval.
        budget = await pacing.get_budget(db_session)
        budget.dm_sent_count = get_settings().dm_daily_cap
        await db_session.flush()

        outcome = await draft_agent.approve_and_send(db_session, client, record=record)

    assert outcome.status == OutreachStatus.FAILED.value
    assert "cap" in (outcome.error or "").lower()
    assert not send.called


@pytest.mark.asyncio
@respx.mock
async def test_draft_is_refused_once_the_daily_cap_is_reached(
    db_session, auth_token, fake_llm, x_api_mock
) -> None:
    """The plan requires the draft pre-flight to refuse at the daily cap."""
    from app.config import get_settings

    budget = await pacing.get_budget(db_session)
    budget.dm_sent_count = get_settings().dm_daily_cap
    await db_session.flush()

    async with XClient(db_session) as client:
        with pytest.raises(pacing.DailyCapError):
            await draft_agent.create_draft(db_session, client, x_user_id="u1")
