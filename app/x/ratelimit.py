"""Per-endpoint rate limit budgets for the X API v2.

X enforces two independent windows per endpoint:

* a per-user ceiling per 15 minutes (and often a separate 24-hour ceiling),
* a per-app ceiling per 15 minutes.

This module keeps a local token bucket per endpoint so the client never knowingly
exceeds the per-user ceiling, and it records the ``x-rate-limit-*`` response headers
so a window pause can be scheduled precisely instead of sleeping blindly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

#: Endpoint -> (per-user limit per 15 minutes, per-user limit per 24 hours).
#:
#: Sourced from X's published v2 rate limit reference and cross-checked against
#: `build-docs/x-rules-and-limits.md`. ``None`` means "not published".
ENDPOINT_LIMITS: dict[str, tuple[int | None, int | None]] = {
    "/2/users/search": (900, None),
    "/2/tweets/search/recent": (300, None),
    "/2/tweets/search/all": (300, None),
    "/2/users": (900, None),
    "/2/users/by": (900, None),
    "/2/users/by/username/:username": (900, None),
    "/2/users/:id": (900, None),
    "/2/users/:id/tweets": (900, None),
    "/2/users/:id/followers": (300, None),
    "/2/users/:id/following": (300, None),
    "/2/dm_conversations/with/:participant_id/messages": (15, 1440),
    "/2/dm_events": (15, None),
    "/2/users/:id/dm_conversations": (900, None),
    "/2/dm_conversations/:id/dm_events": (900, None),
}

WINDOW_SECONDS = 15 * 60
DAY_SECONDS = 24 * 60 * 60


@dataclass
class _Bucket:
    """A sliding-window counter for one endpoint."""

    limit: int | None
    window_seconds: int
    timestamps: list[float] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def remaining(self, now: float | None = None) -> int | None:
        """Calls left in the current window, or ``None`` when unlimited/unknown."""
        if self.limit is None:
            return None
        now = now if now is not None else time.monotonic()
        self._prune(now)
        return max(0, self.limit - len(self.timestamps))

    def retry_in(self, now: float | None = None) -> float:
        """Seconds until the next call would fit in the window."""
        if self.limit is None:
            return 0.0
        now = now if now is not None else time.monotonic()
        self._prune(now)
        if len(self.timestamps) < self.limit:
            return 0.0
        oldest = min(self.timestamps)
        return max(0.0, oldest + self.window_seconds - now)

    def record(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._prune(now)
        self.timestamps.append(now)


class RateLimiter:
    """Tracks per-endpoint budgets and coordinates waits between callers."""

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._server_state: dict[str, dict[str, int | None]] = {}
        self._daily: dict[str, int] = {}
        self._daily_used: dict[str, int] = {}

    def _bucket(self, endpoint: str) -> _Bucket:
        if endpoint not in self._buckets:
            per_user, per_day = ENDPOINT_LIMITS.get(endpoint, (None, None))
            self._buckets[endpoint] = _Bucket(
                limit=per_user, window_seconds=WINDOW_SECONDS
            )
            if per_day is not None:
                self._daily[endpoint] = per_day
        return self._buckets[endpoint]

    def _lock(self, endpoint: str) -> asyncio.Lock:
        if endpoint not in self._locks:
            self._locks[endpoint] = asyncio.Lock()
        return self._locks[endpoint]

    # -- server-reported state -------------------------------------------
    def record_headers(self, endpoint: str, headers: Mapping[str, str]) -> None:
        """Store the ``x-rate-limit-*`` headers returned by X."""
        limit = _header_int(headers, "x-rate-limit-limit")
        remaining = _header_int(headers, "x-rate-limit-remaining")
        reset = _header_int(headers, "x-rate-limit-reset")
        if limit is None and remaining is None and reset is None:
            return
        self._server_state[endpoint] = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset,
        }

    def server_remaining(self, endpoint: str) -> int | None:
        """Remaining calls according to the last response headers."""
        state = self._server_state.get(endpoint)
        if not state:
            return None
        remaining = state.get("remaining")
        return int(remaining) if remaining is not None else None

    def server_reset_at(self, endpoint: str) -> int | None:
        state = self._server_state.get(endpoint)
        if not state:
            return None
        reset = state.get("reset")
        return int(reset) if reset is not None else None

    # -- budget accounting ------------------------------------------------
    def daily_remaining(self, endpoint: str) -> int | None:
        """Remaining 24-hour allowance for an endpoint, or ``None`` if unlimited."""
        cap = self._daily_cap(endpoint)
        if cap is None:
            return None
        return max(0, cap - self._daily_used.get(endpoint, 0))

    def _daily_cap(self, endpoint: str) -> int | None:
        """The published 24-hour ceiling for an endpoint, if there is one."""
        if endpoint not in self._daily:
            _, per_day = ENDPOINT_LIMITS.get(endpoint, (None, None))
            if per_day is not None:
                self._daily[endpoint] = per_day
        return self._daily.get(endpoint)

    def record_daily(self, endpoint: str) -> None:
        """Count one call against the endpoint's 24-hour ceiling."""
        if self._daily_cap(endpoint) is not None:
            self._daily_used[endpoint] = self._daily_used.get(endpoint, 0) + 1

    def daily_used(self, endpoint: str) -> int:
        return self._daily_used.get(endpoint, 0)

    def snapshot(self) -> dict[str, dict[str, int | None]]:
        """A JSON-serialisable view of all tracked budgets."""
        now = time.monotonic()
        endpoints = set(self._buckets) | set(self._daily) | set(self._daily_used)
        return {
            endpoint: {
                "limit_15m": (
                    self._buckets[endpoint].limit if endpoint in self._buckets else None
                ),
                "used_15m": (
                    len(self._buckets[endpoint].timestamps)
                    if endpoint in self._buckets
                    else 0
                ),
                "remaining_15m": (
                    self._buckets[endpoint].remaining(now)
                    if endpoint in self._buckets
                    else None
                ),
                "server_remaining": self.server_remaining(endpoint),
                "server_reset_at": self.server_reset_at(endpoint),
                "daily_limit": self._daily.get(endpoint),
                "daily_used": self._daily_used.get(endpoint, 0),
            }
            for endpoint in sorted(endpoints)
        }

    async def acquire(self, endpoint: str, *, max_wait_seconds: float = 300.0) -> None:
        """Block until a call to ``endpoint`` fits within the local budget.

        Respects the stricter of the local sliding window and the server-reported
        remaining count, so a window pause is scheduled exactly when X says it resets.
        """
        bucket = self._bucket(endpoint)
        lock = self._lock(endpoint)
        deadline = time.monotonic() + max_wait_seconds

        while True:
            async with lock:
                server_remaining = self.server_remaining(endpoint)
                local_remaining = bucket.remaining()

                blocked_by_server = server_remaining is not None and server_remaining <= 0
                blocked_by_local = local_remaining is not None and local_remaining <= 0

                if not blocked_by_server and not blocked_by_local:
                    bucket.record()
                    return

                wait = 0.0
                if blocked_by_local:
                    wait = max(wait, bucket.retry_in())
                if blocked_by_server:
                    reset_at = self.server_reset_at(endpoint)
                    if reset_at:
                        wait = max(wait, max(1.0, reset_at - time.time()))
                    else:
                        wait = max(wait, 15.0)

            if time.monotonic() + wait > deadline:
                raise TimeoutError(
                    f"Rate limit budget for {endpoint} did not free up within "
                    f"{max_wait_seconds:.0f}s"
                )
            await asyncio.sleep(min(wait, 30.0))


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    try:
        raw = headers.get(name)
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
