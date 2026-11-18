"""Route-level tests for every API endpoint.

The X API is mocked with ``respx`` and the Bedrock model is replaced by a
deterministic fake, so the suite exercises routing, validation, dependency
injection and the response models without touching the network.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient

from app.api.auth_routes import router as auth_router
from app.api.compliance import router as compliance_router
from app.api.discovery import router as discovery_router
from app.api.health import router as health_router
from app.api.outreach import router as outreach_router
from app.db import get_session
from app.errors import AppError
from app.models import (
    DiscoveredAccount,
    DiscoveryRun,
    DmEvent,
    OAuthToken,
    OutreachRecord,
    OutreachStatus,
    SuppressionEntry,
)


# ---------------------------------------------------------------------------
# Application fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def app() -> FastAPI:
    """A FastAPI app with only the API routers (no DB lifespan side effects)."""
    application = FastAPI(title="test-app")
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(discovery_router)
    application.include_router(outreach_router)
    application.include_router(compliance_router)

    @application.exception_handler(AppError)
    async def _handle(request, exc: AppError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @application.exception_handler(RequestValidationError)
    async def _handle_validation(request, exc: RequestValidationError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request payload is invalid",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    return application


@pytest_asyncio.fixture
async def client(
    app: FastAPI, db_session
) -> AsyncGenerator[AsyncClient, None]:
    """An ASGI client whose requests share the test's database session."""

    async def _override() -> AsyncGenerator:
        yield db_session

    app.dependency_overrides[get_session] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture
def fake_llm(monkeypatch) -> None:
    """Replace the Bedrock wrapper with a deterministic canned response."""
    payload = {
        "message": (
            "Saw your post about the new IK solver - congrats on shipping that. "
            "We run a robotics meetup in Singapore and would love to have you speak."
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

    class _FakeLLM:
        async def converse_json(self, **kwargs: Any) -> dict[str, Any]:
            return payload

    monkeypatch.setattr("app.agents.drafts.get_llm", lambda: _FakeLLM())


@pytest.fixture
def x_api_mock(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """Mock the read endpoints used by the draft flow."""
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
    respx_mock.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))
    return respx_mock


@pytest.fixture(autouse=True)
def no_pacing_delay(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pacing.next_delay_seconds", lambda settings=None: 0.0)


# ---------------------------------------------------------------------------
# Health & capabilities
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_reports_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_versioned_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "version" in body
    assert body["checks"]["database"]["status"] == "ok"
    # The plan requires token validity and rate-budget state in the health check.
    assert "x_token" in body["checks"]
    assert "rate_budget" in body["checks"]
    assert body["checks"]["rate_budget"]["dm_daily_cap"] > 0


@pytest.mark.asyncio
async def test_health_reports_the_stored_token(
    client: AsyncClient, db_session
) -> None:
    from datetime import UTC, datetime, timedelta

    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="a",
            refresh_token="r",
            scopes="tweet.read",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.flush()

    response = await client.get("/api/v1/health")
    token_check = response.json()["checks"]["x_token"]
    assert token_check["status"] == "ok"
    assert token_check["x_user_id"] == "self"
    assert token_check["fresh"] is True


@pytest.mark.asyncio
async def test_capabilities_endpoint(client: AsyncClient, auth_token, respx_mock) -> None:
    respx_mock.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(
            200, json={"data": {"id": "self", "username": "operator"}}
        )
    )
    respx_mock.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx_mock.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )

    response = await client.get("/api/v1/x/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["user_context_authenticated"] is True
    assert body["recent_search"] is True
    assert "rate_budget" in body


@pytest.mark.asyncio
async def test_capabilities_report_a_missing_full_archive_tier(
    client: AsyncClient, auth_token, respx_mock
) -> None:
    """The probe must fall back to recent search when /all is not entitled."""
    respx_mock.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(
            200, json={"data": {"id": "self", "username": "operator"}}
        )
    )
    respx_mock.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx_mock.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(
            403,
            json={
                "errors": [
                    {
                        "title": "Forbidden",
                        "detail": "Full archive search is not enabled",
                        "type": "about:blank",
                    }
                ]
            },
        )
    )

    response = await client.get("/api/v1/x/capabilities?force=true")
    assert response.status_code == 200
    body = response.json()
    assert body["full_archive_search"] is False
    assert any("full-archive" in note for note in body["notes"])


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_oauth_start_returns_an_authorization_url(client: AsyncClient) -> None:
    response = await client.get("/auth/x/start")
    assert response.status_code == 200
    body = response.json()
    assert body["authorization_url"].startswith("https://x.com/i/oauth2/authorize")
    assert "code_challenge" in body["authorization_url"]
    assert "dm.write" in body["scopes"]


@pytest.mark.asyncio
async def test_oauth_status_reports_not_connected(client: AsyncClient) -> None:
    response = await client.get("/auth/x/status")
    assert response.status_code == 200
    assert response.json() == {"connected": False}


@pytest.mark.asyncio
async def test_oauth_status_reports_a_stored_token(
    client: AsyncClient, db_session
) -> None:
    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="a",
            refresh_token="r",
            scopes="tweet.read dm.read dm.write",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.flush()

    response = await client.get("/auth/x/status")
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert body["x_user_id"] == "self"
    assert body["fresh"] is True


@pytest.mark.asyncio
async def test_oauth_disconnect_clears_the_token(client: AsyncClient, db_session) -> None:
    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="a",
            refresh_token="r",
            scopes="tweet.read",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.flush()

    response = await client.get("/auth/x/disconnect")
    assert response.status_code == 200
    # The frontend parses the response as JSON, so it must not be a redirect
    # to the HTML docs page.
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Disconnected"}

    assert await db_session.get(OAuthToken, "self") is None


@pytest.mark.asyncio
async def test_oauth_disconnect_clears_all_operator_data(
    client: AsyncClient, db_session
) -> None:
    """Disconnect removes every trace of the operator except the opt-outs.

    The permanent opt-out list and the daily rate budget survive, so people
    who opted out stay excluded and the daily caps cannot be reset.
    """
    from sqlalchemy import select

    from app.models import (
        DiscoveredAccount,
        DiscoveryRun,
        DmConversation,
        DmEvent,
        OAuthState,
        OutreachRecord,
        RateBudget,
        SuppressionEntry,
    )

    run = DiscoveryRun(
        id=uuid.uuid4(),
        input_type="seed_accounts",
        target_segment=["robotics"],
        seed_accounts=["https://x.com/alice"],
        status="COMPLETED",
        stats={},
    )
    db_session.add(run)
    db_session.add(
        OAuthToken(
            x_user_id="self",
            access_token="a",
            refresh_token="r",
            scopes="tweet.read",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    db_session.add(OAuthState(state="s1", code_verifier="v"))
    db_session.add(
        DiscoveredAccount(
            x_user_id="u1",
            username="alice",
            source_run_id=run.id,
        )
    )
    db_session.add(
        OutreachRecord(x_user_id="u1", handle="alice", status="DRAFT")
    )
    db_session.add(DmConversation(x_user_id="u1", conversation_id="c1"))
    db_session.add(
        DmEvent(id="e1", conversation_id="c1", x_user_id="u1", event_type="MessageCreate")
    )
    # These two must survive the disconnect.
    db_session.add(SuppressionEntry(x_user_id="u2", handle="bob", reason="asked"))
    db_session.add(
        RateBudget(day=datetime.now(UTC).strftime("%Y-%m-%d"), dm_sent_count=7)
    )
    await db_session.flush()

    response = await client.get("/auth/x/disconnect")
    assert response.status_code == 200

    assert await db_session.get(OAuthToken, "self") is None
    for model in (
        OAuthState,
        DiscoveredAccount,
        OutreachRecord,
        DmConversation,
        DmEvent,
        DiscoveryRun,
    ):
        remaining = await db_session.execute(select(model))
        assert remaining.scalars().all() == [], model.__tablename__

    # The opt-out list and the daily budget are kept.
    assert await db_session.get(SuppressionEntry, "u2") is not None
    budget_result = await db_session.execute(select(RateBudget))
    budgets = budget_result.scalars().all()
    assert len(budgets) == 1
    assert budgets[0].dm_sent_count == 7


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_discovery_search_requires_a_segment_or_seeds(client: AsyncClient) -> None:
    response = await client.post("/api/v1/discovery/search", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_discovery_search_returns_accounts(
    client: AsyncClient, db_session, auth_token, monkeypatch
) -> None:
    """The search is acknowledged at once and its accounts are read afterwards.

    A discovery run waits on the X API and on several LLM inferences, so it
    runs in the background: the request answers ``202`` with the run to poll
    and the accounts show up once the run has finished.
    """
    from app.agents import search as search_agent

    async def _fake_run(session, client_, **kwargs):  # type: ignore[no-untyped-def]
        run = kwargs["run_id"]
        record = await session.get(DiscoveryRun, run)
        if record is not None:
            record.status = "COMPLETED"
            record.stats = {
                "candidates": 0,
                "bots_filtered": 0,
                "returned": 1,
                "seeds_supplied": 1,
                "seeds_resolved": 1,
                "seeds_included": 1,
                "seeds_dropped": 0,
            }
        account = DiscoveredAccount(
            x_user_id="u1",
            username="alice",
            display_name="Alice",
            bio="Robotics engineer",
            followers_count=1200,
            following_count=400,
            tweets_count=900,
            bot_score=0.1,
            bot_verdict="human",
            bot_reasons=[],
            key_info={"summary": "Builds robot arms"},
            segment_tags=["robotics"],
            source_run_id=run,
        )
        session.add(account)
        await session.flush()
        return search_agent.DiscoveryResult(
            run_id=run,
            target_segment=search_agent.TargetSegment(
                keywords=["robotics"], rationale="Because robotics"
            ),
            query_plan=search_agent.QueryPlan(
                user_queries=["robotics"], post_queries=[]
            ),
            accounts=[
                {
                    "x_user_id": "u1",
                    "username": "alice",
                    "display_name": "Alice",
                    "profile_url": "https://x.com/alice",
                    "bio": "Robotics engineer",
                    "followers_count": 1200,
                    "following_count": 400,
                    "created_at": "2020-01-01T00:00:00.000Z",
                    "bot": {
                        "verdict": "human",
                        "score": 0.1,
                        "reasons": [],
                    },
                    "key_information": {
                        "bio": "Robotics engineer",
                        "occupation": "Engineer",
                        "titles": [],
                        "specializations": ["robotics"],
                        "interests": [],
                        "milestones": [],
                        "future_plans": [],
                        "evidence": [],
                    },
                }
            ],
            candidates_seen=1,
            bots_filtered=0,
            api_calls={"users": 1, "posts": 2},
            estimated_cost_usd=0.01,
            capabilities={"tweets_search_all": True},
            warnings=[],
        )

    monkeypatch.setattr("app.services.search_runner.run_discovery", _fake_run)

    response = await client.post(
        "/api/v1/discovery/search", json={"target_segment": ["robotics engineers"]}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "RUNNING"
    run_id = body["run_id"]
    assert body["poll_url"].endswith(run_id)

    # The work runs in the background, so the run is polled until it finishes.
    for _ in range(100):
        await asyncio.sleep(0.01)
        run_response = await client.get(f"/api/v1/discovery/runs/{run_id}")
        if run_response.json()["status"] != "RUNNING":
            break
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "COMPLETED"

    accounts = await client.get("/api/v1/discovery/accounts")
    assert accounts.status_code == 200
    body = accounts.json()
    assert len(body) == 1
    assert body[0]["username"] == "alice"


@pytest.mark.asyncio
async def test_discovery_search_reports_the_seed_accounts(
    client: AsyncClient, db_session, auth_token, monkeypatch
) -> None:
    """The run reports how the seed accounts were treated."""
    from app.agents import search as search_agent

    async def _fake_run(session, client_, **kwargs):  # type: ignore[no-untyped-def]
        run = kwargs["run_id"]
        record = await session.get(DiscoveryRun, run)
        if record is not None:
            record.status = "COMPLETED"
            record.stats = {
                "candidates": 0,
                "bots_filtered": 0,
                "returned": 1,
                "seeds_supplied": 1,
                "seeds_resolved": 1,
                "seeds_included": 1,
                "seeds_dropped": 0,
            }
        await session.flush()
        return search_agent.DiscoveryResult(
            run_id=run,
            target_segment=search_agent.TargetSegment(keywords=[]),
            query_plan=search_agent.QueryPlan(),
            accounts=[
                {
                    "x_user_id": "u1",
                    "username": "alice",
                    "profile_url": "https://x.com/alice",
                    "origin": "seed",
                }
            ],
            candidates_seen=0,
            seeds_included=1,
            seeds_resolved=1,
            seeds_unresolved=0,
            seeds_dropped=0,
        )

    monkeypatch.setattr("app.services.search_runner.run_discovery", _fake_run)

    response = await client.post(
        "/api/v1/discovery/search", json={"seed_accounts": ["https://x.com/alice"]}
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    for _ in range(100):
        await asyncio.sleep(0.01)
        run_response = await client.get(f"/api/v1/discovery/runs/{run_id}")
        if run_response.json()["status"] != "RUNNING":
            break

    stats = run_response.json()["stats"]
    assert stats["seeds_supplied"] == 1
    assert stats["seeds_resolved"] == 1
    assert stats["seeds_included"] == 1
    assert stats["seeds_dropped"] == 0


@pytest.mark.asyncio
async def test_discovery_run_endpoints(client: AsyncClient, db_session) -> None:
    run = DiscoveryRun(
        id=uuid.uuid4(),
        input_type="segment",
        target_segment=["robotics"],
        seed_accounts=[],
        status="completed",
        stats={"candidates": 5},
    )
    db_session.add(run)
    await db_session.flush()

    response = await client.get(f"/api/v1/discovery/runs/{run.id}")
    assert response.status_code == 200
    assert response.json()["stats"] == {"candidates": 5}

    response = await client.get("/api/v1/discovery/runs")
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = await client.get("/api/v1/discovery/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_discovery_account_endpoints(client: AsyncClient, db_session) -> None:
    db_session.add(
        DiscoveredAccount(
            x_user_id="u1",
            username="alice",
            display_name="Alice",
            bio="Robotics",
            followers_count=1200,
            following_count=400,
            tweets_count=10,
            bot_score=0.1,
            bot_verdict="human",
            bot_reasons=[],
            key_info={"summary": "s"},
            segment_tags=["robotics"],
        )
    )
    await db_session.flush()

    response = await client.get("/api/v1/discovery/accounts")
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = await client.get("/api/v1/discovery/accounts?bot_verdict=human")
    assert len(response.json()) == 1

    response = await client.get("/api/v1/discovery/accounts?bot_verdict=likely_bot")
    assert len(response.json()) == 0

    response = await client.get("/api/v1/discovery/accounts/u1")
    assert response.status_code == 200
    assert response.json()["username"] == "alice"

    response = await client.get("/api/v1/discovery/accounts/nope")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_suppressed_accounts_are_removed_from_discovery_results(
    client: AsyncClient, db_session
) -> None:
    """A suppressed account must not show up in the discovery list."""
    db_session.add(
        DiscoveredAccount(
            x_user_id="u1",
            username="alice",
            display_name="Alice",
            bio="Robotics",
            followers_count=1200,
            following_count=400,
            tweets_count=10,
            bot_score=0.1,
            bot_verdict="human",
            bot_reasons=[],
            key_info={},
            segment_tags=[],
        )
    )
    db_session.add(SuppressionEntry(x_user_id="u1", handle="alice"))
    await db_session.flush()

    response = await client.get("/api/v1/discovery/accounts")
    assert response.status_code == 200
    assert response.json() == []

    # The suppression list stays auditable when explicitly requested.
    response = await client.get("/api/v1/discovery/accounts?include_suppressed=true")
    assert len(response.json()) == 1


# ---------------------------------------------------------------------------
# Outreach
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_create_draft_returns_a_pending_draft(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
    response = await client.post(
        "/api/v1/outreach/drafts",
        json={"x_user_id": "u1", "goal": "Invite to a meetup", "tone": "warm"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == OutreachStatus.PENDING_APPROVAL.value
    assert body["draft_versions"][0]["text"]
    assert body["draft_versions"][0]["citations"]


@pytest.mark.asyncio
async def test_create_draft_rejects_a_blank_user_id(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "   "}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_create_draft_conflicts_when_already_contacted(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
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
    await db_session.flush()

    response = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "u1"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_outreach"


@pytest.mark.asyncio
async def test_create_draft_conflicts_when_suppressed(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
    db_session.add(SuppressionEntry(x_user_id="u1", handle="alice"))
    await db_session.flush()

    response = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "u1"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "suppressed"


@pytest.mark.asyncio
@respx.mock
async def test_approve_draft_sends_the_message(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
    created = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "u1"}
    )
    record_id = created.json()["id"]

    response = await client.post(f"/api/v1/outreach/drafts/{record_id}/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == OutreachStatus.SENT.value
    assert body["x_message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_approve_unknown_draft_is_404(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/outreach/drafts/00000000-0000-0000-0000-000000000000/approve"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_approve_with_queued_dispatch_marks_the_record_approved(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
    """The queued mode hands the record to the drip dispatcher."""
    created = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "u1"}
    )
    record_id = created.json()["id"]

    response = await client.post(
        f"/api/v1/outreach/drafts/{record_id}/approve?dispatch=queued"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == OutreachStatus.APPROVED.value
    # Nothing has been sent yet: the dispatcher sends it.
    assert body["x_message_id"] is None


@pytest.mark.asyncio
async def test_resume_dispatch_endpoint_reports_the_queue(
    client: AsyncClient, db_session, monkeypatch
) -> None:
    """The resume endpoint restarts the dispatcher and drains the queue."""
    from app.services import dispatcher as dispatcher_service

    async def _fake_resume(settings=None):  # type: ignore[no-untyped-def]
        return 3

    monkeypatch.setattr(dispatcher_service, "resume", _fake_resume)

    response = await client.post("/api/v1/outreach/dispatch/resume")
    assert response.status_code == 200
    body = response.json()
    assert body["sent"] == 3
    assert "dispatcher_running" in body


@pytest.mark.asyncio
@respx.mock
async def test_deny_draft_creates_a_revision(
    client: AsyncClient, db_session, auth_token, fake_llm, x_api_mock
) -> None:
    created = await client.post(
        "/api/v1/outreach/drafts", json={"x_user_id": "u1"}
    )
    record_id = created.json()["id"]

    response = await client.post(
        f"/api/v1/outreach/drafts/{record_id}/deny",
        json={"feedback": "too formal"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["draft_versions"]) == 2
    assert body["draft_versions"][1]["feedback"] == "too formal"
    assert body["status"] == OutreachStatus.PENDING_APPROVAL.value


@pytest.mark.asyncio
async def test_deny_requires_feedback(client: AsyncClient, db_session) -> None:
    record = OutreachRecord(
        id=uuid.uuid4(),
        x_user_id="u1",
        status=OutreachStatus.PENDING_APPROVAL.value,
        draft_versions=[{"version": 1, "text": "hi"}],
        evidence_snapshot={},
        feedback=[],
    )
    db_session.add(record)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/outreach/drafts/{record.id}/deny", json={"feedback": ""}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_draft_listing_and_queue(client: AsyncClient, db_session) -> None:
    db_session.add(
        OutreachRecord(
            id=uuid.uuid4(),
            x_user_id="u1",
            status=OutreachStatus.PENDING_APPROVAL.value,
            draft_versions=[{"version": 1, "text": "hello"}],
            evidence_snapshot={},
            feedback=[],
        )
    )
    await db_session.flush()

    response = await client.get("/api/v1/outreach/drafts")
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = await client.get("/api/v1/outreach/drafts?status=PENDING_APPROVAL")
    assert len(response.json()) == 1

    response = await client.get("/api/v1/outreach/drafts?status=SENT")
    assert len(response.json()) == 0

    response = await client.get("/api/v1/outreach/queue")
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_budget_endpoint(client: AsyncClient, db_session) -> None:
    response = await client.get("/api/v1/outreach/budget")
    assert response.status_code == 200
    body = response.json()
    assert body["dm_sent"] == 0
    assert body["dm_remaining"] == body["dm_daily_cap"]
    assert body["pacing_delay_seconds"][0] < body["pacing_delay_seconds"][1]


@pytest.mark.asyncio
async def test_conversation_endpoint(client: AsyncClient, db_session) -> None:
    db_session.add(
        DmEvent(
            id="evt-1",
            conversation_id="c1",
            x_user_id="u1",
            sender_id="u1",
            text="hi there",
            event_type="MessageCreate",
        )
    )
    await db_session.flush()

    response = await client.get("/api/v1/outreach/conversations/u1")
    assert response.status_code == 200
    assert response.json()[0]["text"] == "hi there"


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_suppression_crud(client: AsyncClient, db_session) -> None:
    response = await client.post(
        "/api/v1/suppressions",
        json={"x_user_id": "u1", "handle": "alice", "reason": "asked to stop"},
    )
    assert response.status_code == 201
    assert response.json()["handle"] == "alice"

    response = await client.get("/api/v1/suppressions")
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = await client.post("/api/v1/suppressions", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_outreach_record_ledger(client: AsyncClient, db_session) -> None:
    record = OutreachRecord(
        id=uuid.uuid4(),
        x_user_id="u1",
        handle="alice",
        status=OutreachStatus.SENT.value,
        draft_versions=[],
        evidence_snapshot={},
        feedback=["nice"],
    )
    db_session.add(record)
    await db_session.flush()

    response = await client.get("/api/v1/outreach/records")
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = await client.get("/api/v1/outreach/records?status=SENT")
    assert len(response.json()) == 1

    response = await client.get("/api/v1/outreach/records?x_user_id=nobody")
    assert len(response.json()) == 0

    response = await client.get(f"/api/v1/outreach/records/{record.id}")
    assert response.status_code == 200
    assert response.json()["handle"] == "alice"

    response = await client.get(
        "/api/v1/outreach/records/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
