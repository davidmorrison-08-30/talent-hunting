"""OAuth 2.0 PKCE routes for connecting the service to an X account."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.errors import ConfigurationError
from app.x.auth import (
    build_authorize_url,
    code_challenge_for,
    consume_oauth_state,
    exchange_code_for_token,
    generate_code_verifier,
    store_oauth_state,
)

router = APIRouter(prefix="/auth/x", tags=["auth"])


@router.get("/start", summary="Begin the X OAuth 2.0 PKCE flow")
async def start(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Return the X authorization URL the operator must open in a browser."""
    if not settings.x_client_id or not settings.x_client_secret:
        raise ConfigurationError(
            "X_CLIENT_ID and X_CLIENT_SECRET must be set before starting the OAuth flow"
        )

    state = secrets.token_urlsafe(32)
    verifier = generate_code_verifier()
    await store_oauth_state(session, state, verifier)

    authorize_url = build_authorize_url(settings, state, code_challenge_for(verifier))
    return {
        "authorization_url": authorize_url,
        "state": state,
        "redirect_uri": settings.x_redirect_uri,
        "scopes": settings.x_scopes.split(),
    }


@router.get("/callback", summary="Complete the X OAuth 2.0 PKCE flow")
async def callback(
    code: str = Query(..., description="Authorization code returned by X"),
    state: str = Query(..., description="Opaque state value from /auth/x/start"),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Exchange the authorization code for a token and store it."""
    verifier = await consume_oauth_state(session, state)
    token = await exchange_code_for_token(
        session, code=code, code_verifier=verifier, settings=settings
    )

    return HTMLResponse(
        content=(
            "<html><body style='font-family:sans-serif;padding:2rem'>"
            "<h2>X account connected</h2>"
            f"<p>Authorized user id: <code>{token.x_user_id}</code></p>"
            "<p>You can close this window and return to the API.</p>"
            "</body></html>"
        )
    )


@router.get("/status", summary="Show whether an X token is stored")
async def status(
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Report whether the service currently holds a usable X access token."""
    from sqlalchemy import select

    from app.models import OAuthToken
    from app.x.auth import token_is_fresh

    result = await session.execute(select(OAuthToken).limit(1))
    token = result.scalar_one_or_none()
    if token is None:
        return {"connected": False}

    return {
        "connected": True,
        "x_user_id": token.x_user_id,
        "x_username": token.x_username,
        "scopes": token.scopes.split(),
        "expires_at": token.expires_at.isoformat() if token.expires_at else None,
        "fresh": token_is_fresh(token),
        "has_refresh_token": bool(token.refresh_token),
    }


@router.get("/disconnect", summary="Remove the stored X token", response_model=None)
async def disconnect(
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    """Delete the stored token so a different account can be connected."""
    from sqlalchemy import delete

    from app.models import OAuthToken

    await session.execute(delete(OAuthToken))
    return JSONResponse(status_code=200, content={"detail": "Disconnected"})
