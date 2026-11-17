"""Tests for the X API endpoint and parameter contracts.

X has split the user-lookup endpoints and enforces per-endpoint ``max_results``
minimums. These tests lock in the paths and bounds the client uses so an X API
change surfaces as a failing test rather than an opaque HTTP 400 at runtime.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.x.client import (
    SEARCH_POSTS_MIN,
    USER_TIMELINE_MIN,
    XClient,
)
from app.x.ratelimit import ENDPOINT_LIMITS


@pytest.mark.asyncio
@respx.mock
async def test_get_users_by_ids_uses_the_by_ids_endpoint(
    db_session, auth_token
) -> None:
    """Batch hydration must call GET /2/users, not GET /2/users/by.

    ``/2/users/by`` is the by-usernames endpoint and rejects an ``ids``
    parameter with HTTP 400.
    """
    route = respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "1"}]})
    )
    by_usernames = respx.get("https://api.x.com/2/users/by").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    async with XClient(db_session) as client:
        users = await client.get_users_by_ids(["1", "2", "3"])

    assert by_usernames.call_count == 0
    assert route.call_count == 1
    assert users == [{"id": "1"}]
    assert route.calls[0].request.url.params["ids"] == "1,2,3"


@pytest.mark.asyncio
@respx.mock
async def test_get_users_by_ids_caps_a_batch_at_one_hundred(
    db_session, auth_token
) -> None:
    """X accepts at most 100 IDs per request."""
    route = respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    async with XClient(db_session) as client:
        await client.get_users_by_ids([str(i) for i in range(150)])

    sent = route.calls[0].request.url.params["ids"].split(",")
    assert len(sent) == 100


@pytest.mark.asyncio
@respx.mock
async def test_get_users_by_ids_skips_the_call_for_an_empty_batch(
    db_session, auth_token
) -> None:
    """No IDs means no request — X rejects an empty parameter."""
    route = respx.get("https://api.x.com/2/users").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    async with XClient(db_session) as client:
        assert await client.get_users_by_ids([]) == []

    assert route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_search_posts_raises_max_results_to_the_minimum(
    db_session, auth_token
) -> None:
    """X rejects max_results below 10 on both tweet-search endpoints."""
    recent = respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    full_archive = respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )

    async with XClient(db_session) as client:
        await client.search_posts("lang:en", max_results=1)
        await client.search_posts("lang:en", max_results=1, full_archive=True)

    assert recent.calls[0].request.url.params["max_results"] == str(SEARCH_POSTS_MIN)
    assert (
        full_archive.calls[0].request.url.params["max_results"]
        == str(SEARCH_POSTS_MIN)
    )


@pytest.mark.asyncio
@respx.mock
async def test_get_user_tweets_raises_max_results_to_the_minimum(
    db_session, auth_token
) -> None:
    """X rejects max_results below 5 on the user timeline endpoint."""
    route = respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    async with XClient(db_session) as client:
        await client.get_user_tweets("1", max_results=1)

    assert route.calls[0].request.url.params["max_results"] == str(USER_TIMELINE_MIN)


@pytest.mark.asyncio
@respx.mock
async def test_search_users_keeps_its_lower_minimum(
    db_session, auth_token
) -> None:
    """``/2/users/search`` genuinely accepts max_results=1."""
    route = respx.get("https://api.x.com/2/users/search").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )

    async with XClient(db_session) as client:
        await client.search_users("robotics", max_results=1)

    assert route.calls[0].request.url.params["max_results"] == "1"


def test_rate_limit_table_registers_the_by_ids_endpoint() -> None:
    """``/2/users`` must be budgeted, otherwise it is treated as unlimited."""
    assert ENDPOINT_LIMITS["/2/users"] == (900, None)


@pytest.mark.asyncio
@respx.mock
async def test_path_parameters_are_substituted_not_sent_literally(
    db_session, auth_token
) -> None:
    """A ``:id`` placeholder must never reach the wire.

    ``httpx`` does not expand path templates, so the placeholder was sent
    literally and X rejected it with
    ``The `id` query parameter value [:id] is not valid``.
    """
    route = respx.get(url__regex=r"https://api\.x\.com/2/users/[^/]+/tweets").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    async with XClient(db_session) as client:
        await client.get_user_tweets("3222850370")

    assert route.call_count == 1
    sent_path = route.calls[0].request.url.path
    assert sent_path == "/2/users/3222850370/tweets"
    assert ":id" not in sent_path


@pytest.mark.asyncio
@respx.mock
async def test_dm_send_substitutes_the_participant_id(
    db_session, auth_token
) -> None:
    """The DM-send path template must be expanded before the request."""
    route = respx.post(
        url__regex=r"https://api\.x\.com/2/dm_conversations/with/[^/]+/messages"
    ).mock(return_value=httpx.Response(201, json={"data": {"id": "msg-1"}}))

    async with XClient(db_session) as client:
        await client.send_dm("u-42", "hello")

    assert route.call_count == 1
    sent_path = route.calls[0].request.url.path
    assert sent_path == "/2/dm_conversations/with/u-42/messages"
    assert ":participant_id" not in sent_path


@pytest.mark.asyncio
@respx.mock
async def test_capability_probe_succeeds_with_a_valid_max_results(
    db_session, auth_token
) -> None:
    """The probe must not report "recent search unavailable" for a bad parameter.

    X requires ``max_results`` between 10 and 100 on ``/2/tweets/search/recent``;
    probing with 1 produced a 400 and falsely disabled recent search.
    """
    from app.x.capabilities import capability_probe

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    # Full archive is legitimately unavailable on this tier.
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )

    async with XClient(db_session) as client:
        caps = await capability_probe.probe(client, force=True)

    assert caps.user_context_authenticated is True
    assert caps.recent_search is True
    assert caps.full_archive_search is False


@pytest.mark.asyncio
@respx.mock
async def test_capability_probe_query_is_a_valid_standalone_term(
    db_session, auth_token
) -> None:
    """The probe query must not be a conjunction-required operator alone.

    ``lang:en`` on its own is rejected by X with
    ``is/has/lang cannot be used as a standalone operator``.
    """
    from app.x.capabilities import CAPABILITY_PROBE_QUERY, capability_probe

    respx.get("https://api.x.com/2/users/me").mock(
        return_value=httpx.Response(200, json={"data": {"id": "self"}})
    )
    recent = respx.get("https://api.x.com/2/tweets/search/recent").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {}})
    )
    respx.get("https://api.x.com/2/tweets/search/all").mock(
        return_value=httpx.Response(403, json={"errors": [{"message": "not enabled"}]})
    )

    async with XClient(db_session) as client:
        await capability_probe.probe(client, force=True)

    query = recent.calls[0].request.url.params["query"]
    assert query == CAPABILITY_PROBE_QUERY
    # A conjunction-required operator may not stand alone, and a stopword-only
    # query has no positive clause, so the probe uses one substantive term.
    assert ":" not in query
    assert query.lower() not in {"a", "the", "is", "and", "or"}
