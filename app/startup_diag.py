"""Runtime diagnostics for the application startup path.

The ECS log driver captures the container's stdout, but a container that dies
during the FastAPI lifespan can produce no readable log lines at all. The
helpers here write a small JSON record to stdout *before* each startup stage
and *after* it fails, so a crash is diagnosable from the task's log stream.

The records are plain JSON lines on stdout, one per line, prefixed with
``APP_STARTUP_DIAG`` so they can be filtered out of an application's normal
output.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any

#: Prefix on every diagnostic line, so the lines can be grepped.
PREFIX = "APP_STARTUP_DIAG"

#: The wall-clock time the process started, for relative timings.
_STARTED_AT = time.time()


def record(stage: str, **data: Any) -> None:
    """Write one diagnostic record to stdout, unbuffered.

    The write is flushed immediately: a crash in the next stage must not take
    this record with it.
    """
    payload = {
        "ts": round(time.time() - _STARTED_AT, 3),
        "stage": stage,
        "pid": _safe_pid(),
        **data,
    }
    try:
        sys.stdout.write(f"{PREFIX} {json.dumps(payload, default=str)}\n")
        sys.stdout.flush()
    except Exception:  # pragma: no cover - diagnostics must never raise
        pass


def record_failure(stage: str, exc: BaseException) -> None:
    """Write a diagnostic record for a startup stage that raised."""
    record(
        stage,
        outcome="failure",
        error_type=type(exc).__name__,
        error=str(exc)[:2000],
        traceback=traceback.format_exc()[-3000:],
    )


def _safe_pid() -> int:
    import os

    try:
        return os.getpid()
    except Exception:  # pragma: no cover
        return -1
