"""OAuth 2.0 Authorization Code with PKCE for X (Twitter) API v2.

X's DM endpoints require a user-context token. This module implements the full
PKCE flow:

1. ``/auth/x/start`` builds the authorization URL and stores the ``state`` and
   code verifier.
2. ``/auth/x/callback`` exchanges the ``code`` for an access token (plus a
   refresh token when ``offline.access`` was requested).
3. ``get_or_refresh_token`` transparently refreshes the access token before it
   expires (or on a 401), overwriting the stored refresh token because X refresh
   tokens are single-use.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.errors import ConfigurationError, UpstreamError
from app.models import OAuthState, OAuthToken

log = structlog.get_logger(__name__)

#: Refresh this many seconds before the access token actually expires.
TOKEN_EXPIRY_SKEW_SECONDS = 60


def generate_code_verifier() -> str:
    """Create a PKCE code verifier (RFC 7636, 43-128 chars)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")[:128]


def code_challenge_for(verifier: str) -> str:
    """Derive the S256 code challenge from a verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def build_authorize_url(settings: Settings, state: str, code_challenge: str) -> str:
    """Build the X authorization URL the operator opens in their browser."""
    params = {
        "response_type": "code",
        "client_id": settings.x_client_id,
        "redirect_uri": settings.x_redirect_uri,
        "scope": settings.x_scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{settings.x_oauth_base_url}/authorize?{urlencode(params)}"


async def store_oauth_state(session: AsyncSession, state: str, verifier: str) -> None:
    session.add(OAuthState(state=state, code_verifier=verifier))
    await session.flush()


async def consume_oauth_state(session: AsyncSession, state: str) -> str:
    """Return the code verifier for ``state`` and delete it (single use)."""
    result = await session.execute(select(OAuthState).where(OAuthState.state == state))
    row = result.scalar_one_or_none()
    if row is None:
        raise UpstreamError("OAuth state is unknown or has already been used")
    await session.delete(row)
    await session.flush()
    return row.code_verifier


def _expiry_from(data: dict) -> datetime | None:
    """Compute the absolute expiry from the token response."""
    expires_in = data.get("expires_in")
    if expires_in is None:
        return None
    return datetime.now(UTC) + timedelta(seconds=int(expires_in))


async def _post_token_request(
    settings: Settings, payload: dict[str, str]
) -> dict:
    """POST to the X token endpoint with HTTP Basic client authentication."""
    basic = base64.b64encode(
        f"{settings.x_client_id}:{settings.x_client_secret}".encode()
    ).decode()
    async with httpx.AsyncClient(timeout=settings.x_timeout_seconds) as client:
        response = await client.post(
            settings.x_token_url,
            data=payload,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    if response.status_code >= 400:
        log.error(
            "oauth_token_request_failed",
            status=response.status_code,
            body=response.text[:400],
        )
        raise UpstreamError(
            "X token request failed; re-authorization may be required",
            details={"status_code": response.status_code, "body": response.text[:400]},
        )
    return response.json()


async def exchange_code_for_token(
    session: AsyncSession,
    *,
    code: str,
    code_verifier: str,
    settings: Settings | None = None,
) -> OAuthToken:
    """Exchange an authorization ``code`` for an access/refresh token pair."""
    settings = settings or get_settings()
    if not settings.x_client_id or not settings.x_client_secret:
        raise ConfigurationError("X_CLIENT_ID and X_CLIENT_SECRET must be configured")

    data = await _post_token_request(
        settings,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.x_redirect_uri,
            "client_id": settings.x_client_id,
            "code_verifier": code_verifier,
        },
    )

    token = OAuthToken(
        x_user_id=str(data.get("user_id") or data.get("x_user_id") or "unknown"),
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        scopes=data.get("scope", settings.x_scopes),
        expires_at=_expiry_from(data),
    )
    session.add(token)
    await session.flush()
    log.info(
        "oauth_token_stored",
        x_user_id=token.x_user_id,
        has_refresh=bool(token.refresh_token),
    )
    return token


async def refresh_access_token(
    session: AsyncSession,
    token: OAuthToken,
    settings: Settings | None = None,
) -> OAuthToken:
    """Refresh an expired access token using the single-use refresh token."""
    settings = settings or get_settings()
    if not token.refresh_token:
        raise ConfigurationError(
            "No refresh token is stored; re-run the OAuth authorization flow",
            details={"x_user_id": token.x_user_id},
        )

    data = await _post_token_request(
        settings,
        {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": settings.x_client_id,
        },
    )

    token.access_token = data["access_token"]
    # X refresh tokens are single-use: the response may carry a new one.
    if data.get("refresh_token"):
        token.refresh_token = data["refresh_token"]
    token.scopes = data.get("scope", token.scopes)
    token.expires_at = _expiry_from(data)
    await session.flush()
    log.info("oauth_token_refreshed", x_user_id=token.x_user_id)
    return token


def token_is_fresh(token: OAuthToken, skew_seconds: int = TOKEN_EXPIRY_SKEW_SECONDS) -> bool:
    """True when the access token can still be used without refreshing."""
    if token.expires_at is None:
        return True
    expires_at = token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at.timestamp() - skew_seconds > time.time()


async def get_or_refresh_token(
    session: AsyncSession,
    settings: Settings | None = None,
) -> OAuthToken:
    """Return a valid access token, refreshing it when needed."""
    settings = settings or get_settings()
    result = await session.execute(select(OAuthToken).limit(1))
    token = result.scalar_one_or_none()
    if token is None:
        raise ConfigurationError(
            "No X OAuth token is stored. Visit /auth/x/start to authorize the application."
        )
    if not token_is_fresh(token):
        token = await refresh_access_token(session, token, settings)
    return token
