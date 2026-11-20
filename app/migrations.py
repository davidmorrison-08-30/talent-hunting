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
import time
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

#: The advisory lock that serializes the migrations.
#:
#: Two containers that boot at the same time both try to apply the pending
#: revisions. Alembic does not guard against that: the loser of the race
#: fails with "Online migration expected to match one row", and the two
#: ``DROP INDEX`` statements can deadlock. An advisory lock makes the second
#: container wait for the first one to finish.
MIGRATION_LOCK_ID = 0x7A1E47

#: How long to wait for the migration lock, in seconds.
MIGRATION_LOCK_TIMEOUT_SECONDS = 120


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
    attempts = 0

    while asyncio.get_running_loop().time() < deadline:
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            # region agent log
            _diag("database_reachable", attempts=attempts)
            # endregion agent log
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts += 1
            # region agent log
            _diag(
                "database_connect_failed",
                attempt=attempts,
                error_type=type(exc).__name__,
                error=str(exc)[:400],
            )
            # endregion agent log
            await asyncio.sleep(DB_WAIT_INTERVAL_SECONDS)

    raise TimeoutError(f"Database did not become reachable within {timeout}s: {last_error}")


# region agent log
def _diag(stage: str, **data: object) -> None:
    """Write a startup diagnostic record to stdout and the debug log."""
    try:
        from app.startup_diag import record

        record(stage, **data)
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass

    try:
        import json
        import time

        payload = {
            "sessionId": "8a49d1",
            "runId": "startup-diagnosis",
            "hypothesisId": "H1",
            "location": "app/migrations.py",
            "message": stage,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(
            "/home/david/Desktop/talent-hunting/.cursor/debug-8a49d1.log",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass


# endregion agent log


async def run_migrations() -> None:
    """Apply all pending Alembic revisions.

    The revisions are applied under a PostgreSQL advisory lock, so two
    containers that boot at the same time cannot run the same migration
    twice. Without the lock they race: one of them finds the ``alembic_version``
    row already replaced and fails with "Online migration expected to match
    one row ... 0 found", or the two ``DROP INDEX`` statements deadlock. Either
    failure used to stop the container from starting at all.

    Raises:
        Exception: any Alembic or database error, so a failed migration
            prevents the service from starting with a broken schema.
    """

    # region agent log
    try:
        from app.startup_diag import record

        record("alembic_upgrade_start", target="head")
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass
    # endregion agent log

    try:
        await _apply_under_lock()
    except Exception as exc:
        # region agent log
        try:
            from app.startup_diag import record_failure

            record_failure("alembic_upgrade", exc)
        except Exception:  # pragma: no cover - diagnostics must never raise
            pass
        # endregion agent log
        raise

    # region agent log
    try:
        from app.startup_diag import record

        record("alembic_upgrade_ok")
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass
    # endregion agent log


async def _apply_under_lock() -> None:
    """Apply the revisions while holding the migration advisory lock.

    The lock is taken on the application's own engine, so the connection
    belongs to the running event loop. The revisions themselves run in a
    worker thread, because Alembic's ``env.py`` starts its own loop.
    """
    from alembic import command
    from app.db import get_engine

    engine = get_engine()

    async with engine.connect() as connection:
        # A session-level advisory lock is released when the connection
        # closes, so a crashed container does not leave it held.
        #
        # The wait is bounded: if the other container is stuck, this one
        # gives up instead of blocking the startup forever.
        deadline = time.monotonic() + MIGRATION_LOCK_TIMEOUT_SECONDS
        while True:
            got = await connection.execute(
                text(f"SELECT pg_try_advisory_lock({MIGRATION_LOCK_ID})")
            )
            if got.scalar():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Could not acquire the migration lock within "
                    f"{MIGRATION_LOCK_TIMEOUT_SECONDS}s"
                )
            await asyncio.sleep(1.0)

        try:
            await connection.commit()
            await asyncio.to_thread(command.upgrade, _alembic_config(), "head")
        finally:
            await connection.execute(
                text(f"SELECT pg_advisory_unlock({MIGRATION_LOCK_ID})")
            )
            await connection.commit()
