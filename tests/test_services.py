"""Tests for the supporting services: rate limiting, auth, conversations, dispatcher."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.errors import UpstreamError
from app.models import DmConversation, OAuthToken, OutreachStatus
from app.services import conversations as conversation_service
from app.services import dispatcher as dispatcher_service
from app.services import pacing, suppression
from app.x import auth as x_auth
from app.x.client import XClient
from app.x.errors import RateLimitedError
from app.x.ratelimit import RateLimiter


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rate_limiter_tracks_the_sliding_window() -> None:
    limiter = RateLimiter()
    endpoint = "/2/users/search"

    for _ in range(3):
        await limiter.acquire(endpoint)

    snapshot = limiter.snapshot()[endpoint]
    assert snapshot["used_15m"] == 3
    assert snapshot["remaining_15m"] == 897
    assert snapshot["limit_15m"] == 900


@pytest.mark.asyncio
async def test_rate_limiter_respects_server_headers() -> None:
    limiter = RateLimiter()
    endpoint = "/2/users/search"

    limiter.record_headers(
        endpoint,
        {
            "x-rate-limit-limit": "900",
            "x-rate-limit-remaining": "0",
            "x-rate-limit-reset": str(int(time.time()) + 5),
        },
    )

    assert limiter.server_remaining(endpoint) == 0
    assert limiter.server_reset_at(endpoint) is not None

    # The server says "no more calls", so acquire must block and time out.
    with pytest.raises(TimeoutError):
        await limiter.acquire(endpoint, max_wait_seconds=0.1)


@pytest.mark.asyncio
async def test_rate_limiter_tracks_the_daily_ceiling() -> None:
    limiter = RateLimiter()
    endpoint = "/2/dm_conversations/with/:participant_id/messages"

    limiter.record_daily(endpoint)
    assert limiter.daily_remaining(endpoint) == 1439
    assert limiter.daily_used(endpoint) == 1


@pytest.mark.asyncio
async def test_rate_limiter_ignores_unknown_endpoints() -> None:
    limiter = RateLimiter()
    await limiter.acquire("/2/some/unknown/endpoint")
    snapshot = limiter.snapshot()["/2/some/unknown/endpoint"]
    assert snapshot["limit_15m"] is None
    assert snapshot["remaining_15m"] is None


@pytest.mark.asyncio
@respx.mock
async def test_client_maps_429_to_a_rate_limit_error(db_session, auth_token) -> None:
    route = respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(
            429,
            json={"errors": [{"title": "Rate Limit Exceeded", "detail": "too many"}]},
            headers={"x-rate-limit-reset": "9999999999"},
        )
    )

    settings = get_settings().model_copy(update={"x_retry_max_wait_seconds": 0.01})

    async with XClient(db_session, settings=settings) as client:
        with pytest.raises(RateLimitedError):
            await client.search_users("robotics")

    # The client retries before giving up, so more than one attempt was made.
    assert route.call_count > 1


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------
def test_pkce_verifier_and_challenge_are_deterministic() -> None:
    verifier = "a" * 64
    challenge = x_auth.code_challenge_for(verifier)
    assert challenge
    assert "=" not in challenge  # base64url without padding
    assert x_auth.code_challenge_for(verifier) == challenge


def test_generate_code_verifier_is_random_and_url_safe() -> None:
    first = x_auth.generate_code_verifier()
    second = x_auth.generate_code_verifier()
    assert first != second
    assert 43 <= len(first) <= 128
    assert all(c.isalnum() or c in "-._~" for c in first)


@pytest.mark.asyncio
async def test_oauth_state_is_single_use(db_session: AsyncSession) -> None:
    state = "state-123"
    await x_auth.store_oauth_state(db_session, state, "verifier-123")

    verifier = await x_auth.consume_oauth_state(db_session, state)
    assert verifier == "verifier-123"

    # A replayed state must not yield a verifier again.
    with pytest.raises(UpstreamError):
        await x_auth.consume_oauth_state(db_session, state)


@pytest.mark.asyncio
async def test_token_is_refreshed_when_it_expired(db_session: AsyncSession) -> None:
    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="old",
            refresh_token="refresh-1",
            scopes="tweet.read",
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )
    )
    await db_session.flush()

    with respx.mock:
        respx.post("https://api.x.com/2/oauth2/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": "new",
                    "refresh_token": "refresh-2",
                    "expires_in": 7200,
                    "scope": "tweet.read",
                },
            )
        )

        token = await x_auth.get_or_refresh_token(db_session)

    assert token.access_token == "new"
    assert token.refresh_token == "refresh-2"
    assert token.expires_at is not None
    assert token.expires_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_fresh_token_is_returned_without_a_network_call(
    db_session: AsyncSession,
) -> None:
    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="still-good",
            refresh_token="refresh-1",
            scopes="tweet.read",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.flush()

    token = await x_auth.get_or_refresh_token(db_session)
    assert token.access_token == "still-good"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_poll_events_caches_inbound_messages(db_session, auth_token) -> None:
    respx.get("https://api.x.com/2/dm_events").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "evt-1",
                        "event_type": "MessageCreate",
                        "conversation_id": "conv-1",
                        "sender_id": "u1",
                        "text": "Hey, is the meetup still on?",
                        "created_at": "2026-01-01T10:00:00.000Z",
                    }
                ],
                "conversations": [
                    {"id": "conv-1", "participant_ids": ["u1", "self"]}
                ],
            },
        )
    )

    async with XClient(db_session) as client:
        outcome = await conversation_service.poll_events(db_session, client)

    assert outcome.polled is True
    assert outcome.new_events == 1
    assert outcome.conversations_updated == 1
    assert len(outcome.inbound_messages) == 1
    assert outcome.inbound_messages[0].x_user_id == "u1"
    assert outcome.inbound_messages[0].text == "Hey, is the meetup still on?"

    # A second poll must not duplicate the cached event.
    async with XClient(db_session) as client:
        second = await conversation_service.poll_events(db_session, client)
    assert second.new_events == 0

    history = await conversation_service.conversation_history(db_session, "u1")
    assert len(history) == 1
    assert history[0]["text"] == "Hey, is the meetup still on?"


@pytest.mark.asyncio
@respx.mock
async def test_poll_events_records_an_upstream_error(db_session, auth_token) -> None:
    respx.get("https://api.x.com/2/dm_events").mock(
        return_value=httpx.Response(
            403,
            json={"errors": [{"title": "Forbidden", "detail": "missing dm.read"}]},
        )
    )

    async with XClient(db_session) as client:
        outcome = await conversation_service.poll_events(db_session, client)

    assert outcome.polled is False
    assert outcome.errors


@pytest.mark.asyncio
async def test_mark_conversation_read_clears_the_unread_flag(
    db_session: AsyncSession,
) -> None:
    db_session.add(
        DmConversation(x_user_id="u1", conversation_id="c1", unread_inbound=True)
    )
    await db_session.flush()

    await conversation_service.mark_conversation_read(db_session, "u1")

    result = await conversation_service.conversations_needing_reply(db_session)
    assert result == []


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enqueue_marks_the_record_approved(db_session: AsyncSession) -> None:
    from uuid import uuid4

    from app.models import OutreachRecord

    record = OutreachRecord(
        id=uuid4(),
        x_user_id="u1",
        status=OutreachStatus.PENDING_APPROVAL.value,
        draft_versions=[],
        evidence_snapshot={},
        feedback=[],
    )
    db_session.add(record)
    await db_session.flush()

    await dispatcher_service.enqueue(db_session, record.id)

    assert record.status == OutreachStatus.APPROVED.value


@pytest.mark.asyncio
async def test_dispatcher_drains_approved_records(
    db_session: AsyncSession, monkeypatch
) -> None:
    """The dispatcher sends APPROVED records one at a time."""
    from uuid import uuid4

    from app.models import OutreachRecord

    sent: list[str] = []

    async def _fake_send(session, client, *, record, settings=None, apply_pacing_delay=True):  # type: ignore[no-untyped-def]
        sent.append(record.x_user_id)
        record.status = OutreachStatus.SENT.value
        await session.flush()
        return dispatcher_service.draft_agent.SendOutcome(
            record_id=record.id,
            x_user_id=record.x_user_id,
            status=OutreachStatus.SENT.value,
        )

    monkeypatch.setattr(dispatcher_service.draft_agent, "approve_and_send", _fake_send)
    monkeypatch.setattr(
        dispatcher_service, "get_sessionmaker", lambda: db_session_maker(db_session)
    )

    record = OutreachRecord(
        id=uuid4(),
        x_user_id="u1",
        status=OutreachStatus.APPROVED.value,
        draft_versions=[{"version": 1, "text": "hi"}],
        evidence_snapshot={},
        feedback=[],
    )
    db_session.add(record)
    await db_session.flush()

    count = await dispatcher_service.dispatcher._drain_once(_settings())
    assert count == 1
    assert sent == ["u1"]


def db_session_maker(session: AsyncSession):  # type: ignore[no-untyped-def]
    """Build a sessionmaker-like factory bound to the test session."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield session

    return _factory

def _settings():  # type: ignore[no-untyped-def]
    from app.config import get_settings

    return get_settings()


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pacing_delay_is_within_the_configured_window(
    db_session: AsyncSession,
) -> None:
    settings = _settings()
    for _ in range(50):
        delay = pacing.next_delay_seconds(settings)
        assert settings.dm_min_delay_seconds <= delay <= settings.dm_max_delay_seconds


@pytest.mark.asyncio
async def test_dm_sent_counter_increments(db_session: AsyncSession) -> None:
    assert await pacing.dm_sent_today(db_session) == 0
    await pacing.record_dm_sent(db_session)
    assert await pacing.dm_sent_today(db_session) == 1


@pytest.mark.asyncio
async def test_the_budget_row_rolls_over_by_day(db_session: AsyncSession) -> None:
    budget = await pacing.get_budget(db_session)
    budget.day = "2000-01-01"
    budget.dm_sent_count = 99
    await db_session.flush()

    fresh = await pacing.get_budget(db_session)
    assert fresh.day != "2000-01-01"
    assert fresh.dm_sent_count == 0


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_suppress_cancels_a_pending_draft(db_session: AsyncSession) -> None:
    from uuid import uuid4

    from app.models import OutreachRecord

    record = OutreachRecord(
        id=uuid4(),
        x_user_id="u1",
        status=OutreachStatus.PENDING_APPROVAL.value,
        draft_versions=[],
        evidence_snapshot={},
        feedback=[],
    )
    db_session.add(record)
    await db_session.flush()

    await suppression.suppress(db_session, x_user_id="u1", handle="alice")

    assert record.status == OutreachStatus.OPTED_OUT.value

    from app.services import dedup

    assert await dedup.is_suppressed(db_session, "u1") is True

    listed = await suppression.list_suppressions(db_session)
    assert len(listed) == 1
    assert listed[0].x_user_id == "u1"
