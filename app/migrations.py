"""Database schema migrations.

The service exposes a ``RUN_MIGRATIONS_ON_STARTUP`` switch. When it is on,
the FastAPI lifespan applies the pending Alembic revisions before the app
starts serving traffic, so a fresh ECS task definition comes up against an
empty database without a manual ``alembic upgrade head`` step.

Alembic's own ``env.py`` already knows how to run against the async engine,
so the upgrade is driven through ``alembic.command.upgrade`` in a worker
thread. Running it in a thread keeps the event loop free while the schema is
being written, and keeps the ``asyncio.run()`` call inside ``env.py`` from
colliding with the already-running loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic.config import Config
from sqlalchemy import text

from app.db import get_engine

#: Repository root (the directory that holds ``alembic.ini``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: How long to wait for the database to accept connections, in seconds.
DB_WAIT_TIMEOUT_SECONDS = 120

#: Delay between connection attempts, in seconds.
DB_WAIT_INTERVAL_SECONDS = 2.0


def _alembic_config() -> Config:
    """Build an Alembic config pointing at the packaged migrations."""
    ini_path = PROJECT_ROOT / "alembic.ini"
    config = Config(str(ini_path))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


async def wait_for_database(timeout: float = DB_WAIT_TIMEOUT_SECONDS) -> None:
    """Block until the database accepts connections.

    On ECS the API container and the Postgres container start together, so the
    database is often not listening yet when the API boots. Without this wait
    the first migration attempt fails and the task crash-loops.

    Raises:
        TimeoutError: the database did not become reachable in time.
    """
    engine = get_engine()
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: Exception | None = None

    while asyncio.get_running_loop().time() < deadline:
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(DB_WAIT_INTERVAL_SECONDS)

    raise TimeoutError(f"Database did not become reachable within {timeout}s: {last_error}")


async def run_migrations() -> None:
    """Apply all pending Alembic revisions.

    Raises:
        Exception: any Alembic or database error, so a failed migration
            prevents the service from starting with a broken schema.
    """
    from alembic import command

    await asyncio.to_thread(command.upgrade, _alembic_config(), "head")
