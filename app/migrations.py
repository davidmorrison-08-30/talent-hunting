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

from alembic import command

#: Repository root (the directory that holds ``alembic.ini``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _alembic_config() -> Config:
    """Build an Alembic config pointing at the packaged migrations."""
    ini_path = PROJECT_ROOT / "alembic.ini"
    config = Config(str(ini_path))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


async def run_migrations() -> None:
    """Apply all pending Alembic revisions.

    Raises:
        Exception: any Alembic or database error, so a failed migration
            prevents the service from starting with a broken schema.
    """
    await asyncio.to_thread(command.upgrade, _alembic_config(), "head")
