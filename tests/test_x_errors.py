"""Tests for structured X API error logging.

These cover the ``XApiError.log_fields()`` helper and the guarantee that a
failed X call logs the failing endpoint and the upstream response body rather
than collapsing to a generic ``[400] X API request failed`` string.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from structlog.testing import capture_logs

from app.agents.queries import QueryPlan
from app.agents.search import SearchLimits, _search_candidates
from app.x.client import XClient
from app.x.errors import XApiError, x_error_fields

#: A realistic X API 400 body.
X_400_BODY: dict[str, Any] = {
    "errors": [
        {
            "title": "Invalid Request",
            "detail": "Unexpected operator: min_faves",
            "message": "bad query",
        }
    ],
    "status": 400,
}


def test_log_fields_exposes_endpoint_status_and_payload() -> None:
    exc = XApiError(
        "X API request failed",
        status_code=400,
        payload=X_400_BODY,
        endpoint="/2/tweets/search/recent",
    )

    fields = exc.log_fields()

    assert fields["endpoint"] == "/2/tweets/search/recent"
    assert fields["status_code"] == 400
    # The `error` field keeps its previous shape so existing greps still work.
    assert fields["error"] == "[400] X API request failed"
    # `detail` has priority over `message` and `title`.
    assert fields["x_error"] == "Unexpected operator: min_faves"
    assert "Unexpected operator: min_faves" in fields["x_payload"]


def test_log_fields_truncates_a_large_payload() -> None:
    exc = XApiError(
        "X API request failed",
        status_code=402,
        payload={"detail": "x" * 5000},
        endpoint="/2/tweets/search/recent",
    )

    rendered = exc.log_fields()["x_payload"]

    assert rendered is not None
    assert rendered.endswith("...(truncated)")
    assert len(rendered) < 5000


def test_log_fields_reports_no_payload_as_none() -> None:
    fields = XApiError("boom", status_code=500, endpoint="/2/users/me").log_fields()

    assert fields["x_error"] is None
    assert fields["x_payload"] is None
    assert fields["error"] == "[500] boom"


def test_log_fields_handles_a_flat_error_body() -> None:
    fields = XApiError("boom", payload={"detail": "flat detail"}).log_fields()

    assert fields["x_error"] == "flat detail"


def test_x_error_fields_falls_back_for_non_x_exceptions() -> None:
    assert x_error_fields(ValueError("boom")) == {"error": "boom"}


def test_x_error_fields_delegates_to_x_api_errors() -> None:
    exc = XApiError(
        "X API request failed",
        status_code=400,
        payload=X_400_BODY,
        endpoint="/2/tweets/search/recent",
    )

    fields = x_error_fields(exc)

    assert fields["endpoint"] == "/2/tweets/search/recent"
    assert fields["status_code"] == 400
    assert fields["x_error"] == "Unexpected operator: min_faves"


@pytest.mark.asyncio
@respx.mock
async def test_a_failed_search_logs_the_endpoint_and_payload(
    db_session, auth_token, monkeypatch
) -> None:
    """A 400 from tweet search must surface the endpoint and X's own detail."""
    respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(400, json=X_400_BODY)
    )

    with capture_logs() as logs:
        async with XClient(db_session) as client:
            _, _, _, _, failures = await _search_candidates(
                client,
                QueryPlan(
                    user_queries=["robotics engineer"],
                    post_queries=['"robotics" -is:retweet'],
                ),
                SearchLimits(max_accounts=10, max_candidates=10),
                full_archive=False,
            )

    failures_logged = [entry for entry in logs if entry.get("event") == "post_search_failed"]

    assert failures_logged, "expected a post_search_failed log entry"
    failure = failures_logged[0]
    assert failure["endpoint"] == "/2/tweets/search/recent"
    assert failure["status_code"] == 400
    assert failure["x_error"] == "Unexpected operator: min_faves"
    assert "Unexpected operator: min_faves" in failure["x_payload"]
    assert failure["query"] == '"robotics" -is:retweet'

    # The failure must also be surfaced to the caller, not only to the log.
    assert any("Tweet search failed" in message for message in failures)
