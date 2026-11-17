"""Health and capability endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.x.capabilities import capability_probe
from app.x.client import XClient

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/health", summary="Service health with dependency checks")
async def detailed_health(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:  # pragma: no cover - depends on live DB
        checks["database"] = {"status": "error", "detail": str(exc)}

    checks["bedrock"] = {
        "status": "configured" if settings.aws_bedrock_model_id else "missing_model_id",
        "model_id": settings.aws_bedrock_model_id or None,
        "region": settings.aws_region,
    }
    checks["x_oauth"] = {
        "status": "configured"
        if settings.x_client_id and settings.x_client_secret
        else "missing_credentials",
        "redirect_uri": settings.x_redirect_uri,
        "scopes": settings.x_scopes.split(),
    }

    # Token validity: is a usable access token stored?
    from sqlalchemy import select

    from app.models import OAuthToken
    from app.x.auth import token_is_fresh

    token_result = await session.execute(select(OAuthToken).limit(1))
    token = token_result.scalar_one_or_none()
    checks["x_token"] = (
        {
            "status": "ok" if token_is_fresh(token) else "expired",
            "x_user_id": token.x_user_id,
            "expires_at": token.expires_at.isoformat() if token.expires_at else None,
            "fresh": token_is_fresh(token),
            "has_refresh_token": bool(token.refresh_token),
        }
        if token is not None
        else {"status": "not_connected"}
    )

    # Rate-budget state: how much of today's budget is used?
    from app.services.pacing import get_budget

    budget = await get_budget(session)
    checks["rate_budget"] = {
        "day": budget.day,
        "dm_sent": budget.dm_sent_count,
        "dm_daily_cap": settings.dm_daily_cap,
        "dm_remaining": max(0, settings.dm_daily_cap - budget.dm_sent_count),
        "posts_read": budget.posts_read_count,
        "users_read": budget.users_read_count,
        "dm_events_polled": budget.dm_events_polled,
        "estimated_cost_usd": round(budget.estimated_cost_usd, 4),
    }

    return {
        "status": "ok",
        "version": settings.app_version,
        "checks": checks,
    }


@router.get(
    "/api/v1/x/capabilities",
    summary="Live probe of the connected X account's endpoint access",
)
async def capabilities(
    force: bool = False,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Report which X endpoints the connected account can actually call.

    X retired the Basic/Pro subscription tiers in 2026, so entitlement depends on
    the live account. This endpoint probes (and caches) the real capabilities
    rather than assuming a plan name.
    """
    async with XClient(session) as client:
        caps = await capability_probe.probe(client, force=force)

    payload = caps.to_payload()
    payload["rate_budget"] = client.limiter.snapshot()
    return payload
