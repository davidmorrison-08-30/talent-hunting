"""Shared pytest fixtures: an isolated database and mocked X API."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest_asyncio

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/talent_hunting_test",
)
os.environ.setdefault("AWS_BEDROCK_MODEL_ID", "test.model")
os.environ.setdefault("X_CLIENT_ID", "test-client-id")
os.environ.setdefault("X_CLIENT_SECRET", "test-client-secret")

from app.db import Base, dispose_engine, get_engine, get_sessionmaker  # noqa: E402
from app.models import OAuthToken  # noqa: E402


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    """A session bound to a schema that is recreated for each test."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    maker = get_sessionmaker()
    async with maker() as session:
        yield session

    await dispose_engine()


@pytest_asyncio.fixture
async def auth_token(db_session) -> OAuthToken:
    """Store a fake OAuth token so the X client can authenticate."""
    from datetime import UTC, datetime, timedelta

    token = OAuthToken(
        x_user_id="self",
        access_token="fake-access-token",
        refresh_token="fake-refresh-token",
        scopes="tweet.read users.read dm.read dm.write offline.access",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
    )
    db_session.add(token)
    await db_session.flush()
    return token
