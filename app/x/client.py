"""Typed async client for the X API v2 endpoints used by this service.

Every request goes through :meth:`XClient.request`, which centralizes:

* OAuth 2.0 user-context bearer authentication (with automatic refresh),
* per-endpoint rate limit budgeting,
* ``x-rate-limit-*`` header tracking,
* retry with backoff on 429 and transient 5xx,
* mapping HTTP status codes onto the typed errors in :mod:`app.x.errors`.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Iterable
from typing import Any

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import OAuthToken
from app.x.auth import get_or_refresh_token
from app.x.errors import (
    AuthExpiredError,
    BlockedDMError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    XApiError,
    XCapabilityUnavailable,
)
from app.x.ratelimit import RateLimiter

log = structlog.get_logger(__name__)

USER_FIELDS = (
    "created_at,description,public_metrics,verified,verified_type,"
    "location,url,entities,profile_image_url"
)
TWEET_FIELDS = "author_id,created_at,public_metrics,lang,entities,conversation_id,referenced_tweets"
TWEET_EXPANSIONS = "author_id,referenced_tweets.id"
USER_EXPANSIONS = "pinned_tweet_id"

#: Endpoints that are only available on higher X access tiers.
TIER_GATED_ENDPOINTS = {"/2/tweets/search/all"}

#: Minimum ``max_results`` accepted by ``/2/tweets/search/recent`` and
#: ``/2/tweets/search/all``. X rejects anything below 10 with HTTP 400.
SEARCH_POSTS_MIN = 10
#: Maximum ``max_results`` accepted by ``/2/tweets/search/recent``.
SEARCH_POSTS_MAX_RECENT = 100
#: Maximum ``max_results`` accepted by ``/2/tweets/search/all``.
SEARCH_POSTS_MAX_FULL_ARCHIVE = 500

#: Minimum ``max_results`` accepted by ``/2/users/:id/tweets``.
USER_TIMELINE_MIN = 5
#: Maximum ``max_results`` accepted by ``/2/users/:id/tweets``.
USER_TIMELINE_MAX = 100

#: Minimum and maximum ``max_results`` accepted by ``/2/users/search``.
USER_SEARCH_MIN = 1
USER_SEARCH_MAX = 1000

#: Minimum and maximum ``max_results`` accepted by ``/2/dm_events`` and the
#: per-conversation DM event endpoints.
DM_EVENTS_MIN = 1
DM_EVENTS_MAX = 100

#: Minimum and maximum ``max_results`` accepted by
#: ``/2/users/:id/dm_conversations``.
DM_CONVERSATIONS_MIN = 1
DM_CONVERSATIONS_MAX = 100


class XClient:
    """Async wrapper around the X API v2."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.limiter = limiter or RateLimiter()
        self._http: httpx.AsyncClient | None = None
        self._token: OAuthToken | None = None
        #: Endpoint -> number of calls made (used for cost/stat reporting).
        self.call_counts: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------
    async def __aenter__(self) -> XClient:
        self._http = httpx.AsyncClient(
            base_url=self.settings.x_api_base_url,
            timeout=self.settings.x_timeout_seconds,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.settings.x_api_base_url,
                timeout=self.settings.x_timeout_seconds,
            )
        return self._http

    # -- core request ------------------------------------------------------
    async def _access_token(self) -> str:
        if self._token is None:
            self._token = await get_or_refresh_token(self.session, self.settings)
        return self._token.access_token

    async def _refresh_token_after_auth_error(self) -> str:
        from app.x.auth import refresh_access_token

        if self._token is None:
            self._token = await get_or_refresh_token(self.session, self.settings)
        else:
            self._token = await refresh_access_token(
                self.session, self._token, self.settings
            )
        return self._token.access_token

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        retry_auth: bool = True,
        path_params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform an authenticated X API request with retries and error mapping.

        ``path_params`` substitutes ``:name`` placeholders in ``endpoint``. The
        placeholder is **not** sent literally — X rejects ``/2/users/:id/tweets``
        with ``The `id` query parameter value [:id] is not valid``.
        """
        attempts = 0
        max_attempts = max(1, self.settings.x_max_retries)

        while attempts < max_attempts:
            attempts += 1
            token = await self._access_token()

            await self.limiter.acquire(endpoint)
            self.call_counts[endpoint] = self.call_counts.get(endpoint, 0) + 1
            self.limiter.record_daily(endpoint)

            response = await self.http.request(
                method,
                _fill_path_params(endpoint, path_params),
                params=params,
                json=json_body,
                headers={"Authorization": f"Bearer {token}"},
            )
            self.limiter.record_headers(endpoint, response.headers)

            if response.status_code == 429:
                reset_at = _int_header(response.headers, "x-rate-limit-reset")
                retry_after = _int_header(response.headers, "retry-after")
                if retry_after is None and reset_at is not None:
                    retry_after = max(1, int(reset_at - time.time()))
                wait = min(float(retry_after or 60), self.settings.x_retry_max_wait_seconds)
                if attempts < max_attempts:
                    log.warning(
                        "x_rate_limited",
                        endpoint=endpoint,
                        attempt=attempts,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RateLimitedError(
                    "X API rate limit exceeded",
                    reset_at=reset_at,
                    endpoint=endpoint,
                )

            if response.status_code == 401:
                if retry_auth:
                    log.info("x_token_expired", endpoint=endpoint)
                    await self._refresh_token_after_auth_error()
                    retry_auth = False
                    continue
                raise AuthExpiredError(
                    "X access token was rejected", status_code=401, endpoint=endpoint
                )

            if response.status_code == 403:
                payload = _safe_json(response)
                if endpoint.endswith("/messages"):
                    raise BlockedDMError(
                        "Recipient does not accept direct messages from this account",
                        status_code=403,
                        payload=payload,
                        endpoint=endpoint,
                    )
                if endpoint in TIER_GATED_ENDPOINTS or _looks_like_tier_error(payload):
                    raise XCapabilityUnavailable(
                        "This X endpoint is not available on the current access tier",
                        status_code=403,
                        payload=payload,
                        endpoint=endpoint,
                    )
                raise ForbiddenError(
                    "X API denied the request",
                    status_code=403,
                    payload=payload,
                    endpoint=endpoint,
                )

            if response.status_code == 404:
                raise NotFoundError(
                    "The requested X resource was not found",
                    status_code=404,
                    payload=_safe_json(response),
                    endpoint=endpoint,
                )

            if response.status_code >= 500:
                if attempts < max_attempts:
                    backoff = min(2**attempts + random.uniform(0, 1), 30)
                    log.warning(
                        "x_upstream_error", endpoint=endpoint, status=response.status_code
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise XApiError(
                    "X API returned a server error",
                    status_code=response.status_code,
                    payload=_safe_json(response),
                    endpoint=endpoint,
                )

            if response.status_code >= 400:
                raise XApiError(
                    "X API request failed",
                    status_code=response.status_code,
                    payload=_safe_json(response),
                    endpoint=endpoint,
                )

            if not response.content:
                return {}
            return response.json()

        raise XApiError("X API request exhausted all retries", endpoint=endpoint)

    # -- users -------------------------------------------------------------
    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Look up a user by handle. Returns ``None`` when not found."""
        try:
            data = await self.request(
                "GET",
                "/2/users/by/username/:username",
                path_params={"username": username},
                params={
                    "username": username,
                    "user.fields": USER_FIELDS,
                    "expansions": USER_EXPANSIONS,
                    "tweet.fields": TWEET_FIELDS,
                },
            )
        except NotFoundError:
            return None
        users = data.get("data") or []
        return users[0] if users else None

    async def get_users_by_ids(self, user_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Batch-lookup up to 100 users by id.

        Uses ``GET /2/users`` (the by-IDs endpoint). ``GET /2/users/by`` is the
        by-usernames endpoint and rejects an ``ids`` parameter.
        """
        ids = [str(uid) for uid in user_ids if uid]
        if not ids:
            return []
        data = await self.request(
            "GET",
            "/2/users",
            params={
                "ids": ",".join(ids[:100]),
                "user.fields": USER_FIELDS,
                "expansions": USER_EXPANSIONS,
                "tweet.fields": TWEET_FIELDS,
            },
        )
        return list(data.get("data") or [])

    async def get_user_tweets(
        self,
        user_id: str,
        *,
        max_results: int = 20,
        pagination_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a user's recent posts, newest first."""
        params: dict[str, Any] = {
            "max_results": max(USER_TIMELINE_MIN, min(max_results, USER_TIMELINE_MAX)),
            "tweet.fields": TWEET_FIELDS,
            "expansions": TWEET_EXPANSIONS,
            "user.fields": USER_FIELDS,
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self.request(
            "GET",
            "/2/users/:id/tweets",
            path_params={"id": user_id},
            params=params,
        )

    async def search_users(
        self, query: str, *, max_results: int = 100, next_token: str | None = None
    ) -> dict[str, Any]:
        """Search for users matching a plain-text query."""
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(USER_SEARCH_MIN, min(max_results, USER_SEARCH_MAX)),
            "user.fields": USER_FIELDS,
            "expansions": USER_EXPANSIONS,
            "tweet.fields": TWEET_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token
        return await self.request("GET", "/2/users/search", params=params)

    # -- posts -------------------------------------------------------------
    async def search_posts(
        self,
        query: str,
        *,
        max_results: int = 100,
        next_token: str | None = None,
        full_archive: bool = False,
    ) -> dict[str, Any]:
        """Search recent posts (7 days) or the full archive."""
        endpoint = "/2/tweets/search/all" if full_archive else "/2/tweets/search/recent"
        max_allowed = (
            SEARCH_POSTS_MAX_FULL_ARCHIVE if full_archive else SEARCH_POSTS_MAX_RECENT
        )
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(SEARCH_POSTS_MIN, min(max_results, max_allowed)),
            "tweet.fields": TWEET_FIELDS,
            "expansions": TWEET_EXPANSIONS,
            "user.fields": USER_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token
        return await self.request("GET", endpoint, params=params)

    # -- direct messages ---------------------------------------------------
    async def list_dm_conversations(
        self, user_id: str, *, max_results: int = 100
    ) -> dict[str, Any]:
        """List DM conversations for the authenticated user."""
        return await self.request(
            "GET",
            "/2/users/:id/dm_conversations",
            path_params={"id": user_id},
            params={
                "max_results": max(
                    DM_CONVERSATIONS_MIN, min(max_results, DM_CONVERSATIONS_MAX)
                )
            },
        )

    async def get_dm_events_for_conversation(
        self, conversation_id: str, *, max_results: int = 100
    ) -> dict[str, Any]:
        """Fetch the event history of one DM conversation."""
        return await self.request(
            "GET",
            "/2/dm_conversations/:id/dm_events",
            path_params={"id": conversation_id},
            params={
                "max_results": max(DM_EVENTS_MIN, min(max_results, DM_EVENTS_MAX)),
                "event_types": ["MessageCreate", "ParticipantsJoin", "ParticipantsLeave"],
                "dm_event.fields": "id,text,event_type,created_at,sender_id,participant_ids",
                "expansions": "sender_id",
                "user.fields": "id,username,name",
            },
        )

    async def get_dm_events_for_participant(
        self, participant_id: str, *, max_results: int = 100
    ) -> dict[str, Any]:
        """Fetch DM events for a one-to-one conversation with a participant."""
        return await self.request(
            "GET",
            "/2/dm_conversations/with/:participant_id/dm_events",
            path_params={"participant_id": participant_id},
            params={
                "max_results": max(DM_EVENTS_MIN, min(max_results, DM_EVENTS_MAX)),
                "event_types": ["MessageCreate", "ParticipantsJoin", "ParticipantsLeave"],
                "dm_event.fields": "id,text,event_type,created_at,sender_id,participant_ids",
                "expansions": "sender_id",
                "user.fields": "id,username,name",
            },
        )

    async def poll_dm_events(
        self, *, max_results: int = 100, next_token: str | None = None
    ) -> dict[str, Any]:
        """Poll the account-wide DM event stream (tight limit: 15 per 15 minutes)."""
        params: dict[str, Any] = {
            "max_results": max(DM_EVENTS_MIN, min(max_results, DM_EVENTS_MAX)),
            "event_types": ["MessageCreate", "ParticipantsJoin", "ParticipantsLeave"],
            "dm_event.fields": "id,text,event_type,created_at,sender_id,participant_ids",
            "expansions": "sender_id",
            "user.fields": "id,username,name",
        }
        if next_token:
            params["next_token"] = next_token
        return await self.request("GET", "/2/dm_events", params=params)

    async def send_dm(self, participant_id: str, text: str) -> dict[str, Any]:
        """Send a direct message to a participant."""
        return await self.request(
            "POST",
            "/2/dm_conversations/with/:participant_id/messages",
            path_params={"participant_id": participant_id},
            json_body={"text": text},
        )

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict[str, int]:
        """Call counts per endpoint for cost reporting."""
        return dict(self.call_counts)


def _fill_path_params(endpoint: str, path_params: dict[str, str] | None) -> str:
    """Substitute ``:name`` placeholders in an endpoint path.

    ``httpx`` does not expand path templates, so a placeholder that reaches the
    wire is sent literally and X rejects it (for example
    ``The `id` query parameter value [:id] is not valid``).
    """
    if not path_params:
        return endpoint

    filled = endpoint
    for name, value in path_params.items():
        filled = filled.replace(f":{name}", str(value))
    return filled


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {"raw": response.text[:500]}
    return data if isinstance(data, dict) else {"data": data}


def _int_header(headers: httpx.Headers, name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _looks_like_tier_error(payload: dict[str, Any]) -> bool:
    """Detect X's 'you do not have access to this endpoint' error shape."""
    text = str(payload).lower()
    return any(
        marker in text
        for marker in (
            "not authorized",
            "does not have access",
            "capability",
            "feature not available",
            "not enabled",
        )
    )
