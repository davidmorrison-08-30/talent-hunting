"""Runtime capability detection for the X API.

X retired the Basic/Pro subscription tiers in 2026 and now sells pay-per-use and
Enterprise access. Which endpoints an account can actually call therefore depends
on the account's live entitlements, not on a static plan name.

This module probes the account once and caches the result so the discovery agent
can pick full-archive search when it is available and fall back to the 7-day
recent search when it is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.x.client import SEARCH_POSTS_MIN, XClient
from app.x.errors import XApiError, XCapabilityUnavailable

log = structlog.get_logger(__name__)

#: How long a probe result is considered fresh.
PROBE_TTL_SECONDS = 3600

#: Probe query used to test whether an X search tier is available.
#:
#: X imposes two hard constraints on a tweet-search query:
#:
#: * a *conjunction-required* operator may not stand alone — ``lang:en`` alone
#:   is rejected with ``is/has/lang cannot be used as a standalone operator``;
#: * the query must contain at least one *positive, non-stopword* clause — a
#:   bare ``a`` is rejected with ``Rules must contain at least one positive,
#:   non-stopword clause``.
#:
#: A single substantive term satisfies both rules, so the probe measures the
#: account's tier rather than the query syntax.
CAPABILITY_PROBE_QUERY = "AI"


@dataclass
class Capabilities:
    """What the connected X account can do right now."""

    full_archive_search: bool = False
    recent_search: bool = True
    dm_read: bool = True
    dm_write: bool = True
    user_context_authenticated: bool = False
    probed_at: float = field(default_factory=time.time)
    notes: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.probed_at

    def to_payload(self) -> dict[str, Any]:
        return {
            "full_archive_search": self.full_archive_search,
            "recent_search": self.recent_search,
            "dm_read": self.dm_read,
            "dm_write": self.dm_write,
            "user_context_authenticated": self.user_context_authenticated,
            "probed_at": self.probed_at,
            "age_seconds": round(self.age_seconds, 1),
            "notes": self.notes,
        }


class CapabilityProbe:
    """Probes and caches the account's X API capabilities."""

    def __init__(self) -> None:
        self._cached: Capabilities | None = None

    def cached(self, *, max_age: float = PROBE_TTL_SECONDS) -> Capabilities | None:
        """Return the cached capabilities if they are still fresh."""
        if self._cached is None:
            return None
        if self._cached.age_seconds > max_age:
            return None
        return self._cached

    def invalidate(self) -> None:
        self._cached = None

    async def probe(self, client: XClient, *, force: bool = False) -> Capabilities:
        """Probe the account's capabilities, using the cached result when possible."""
        if not force:
            cached = self.cached()
            if cached is not None:
                return cached

        caps = Capabilities()
        notes: list[str] = []

        # 1. User-context authentication: a users/me lookup proves the token works.
        try:
            await client.request(
                "GET",
                "/2/users/me",
                params={"user.fields": "id,username"},
            )
            caps.user_context_authenticated = True
        except XApiError as exc:
            detail = exc.log_fields().get("x_error")
            suffix = f" ({detail})" if detail else ""
            notes.append(f"user-context authentication failed{suffix}: {exc}")
            log.warning("capability_auth_failed", **exc.log_fields())

        # 2. Recent search availability.
        try:
            await client.search_posts(
                CAPABILITY_PROBE_QUERY, max_results=SEARCH_POSTS_MIN
            )
            caps.recent_search = True
        except XCapabilityUnavailable as exc:
            caps.recent_search = False
            notes.append(f"recent search unavailable: {exc}")
        except XApiError as exc:
            caps.recent_search = False
            detail = exc.log_fields().get("x_error")
            suffix = f" ({detail})" if detail else ""
            notes.append(f"recent search probe failed{suffix}: {exc}")
            log.warning("capability_recent_search_probe_failed", **exc.log_fields())

        # 3. Full-archive search availability (pay-per-use / Enterprise only).
        try:
            await client.search_posts(
                CAPABILITY_PROBE_QUERY,
                max_results=SEARCH_POSTS_MIN,
                full_archive=True,
            )
            caps.full_archive_search = True
        except XCapabilityUnavailable:
            caps.full_archive_search = False
            notes.append(
                "full-archive search is not enabled on this account; "
                "falling back to the 7-day recent search"
            )
            log.info("capability_full_archive_unavailable")
        except XApiError as exc:
            caps.full_archive_search = False
            detail = exc.log_fields().get("x_error")
            suffix = f" ({detail})" if detail else ""
            notes.append(f"full-archive search probe failed{suffix}: {exc}")
            log.warning("capability_full_archive_probe_failed", **exc.log_fields())

        caps.notes = notes
        self._cached = caps
        log.info(
            "capabilities_probed",
            full_archive=caps.full_archive_search,
            recent=caps.recent_search,
            authenticated=caps.user_context_authenticated,
        )
        return caps


#: Process-wide probe cache shared by the API routers.
capability_probe = CapabilityProbe()
