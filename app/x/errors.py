"""Errors raised by the X API layer."""

from __future__ import annotations

import json
import time
from typing import Any

#: Maximum number of characters of the raw X response body kept in log output.
MAX_PAYLOAD_CHARS = 2000

#: X error keys, in priority order, used to pull a human-readable message out
#: of an X API error body.
_ERROR_MESSAGE_KEYS = ("detail", "message", "title", "reason")


class XApiError(Exception):
    """Base class for X API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
        self.endpoint = endpoint

    def __str__(self) -> str:
        prefix = f"[{self.status_code}]" if self.status_code else ""
        return f"{prefix} {self.message}"

    def log_fields(self, *, max_payload_chars: int = MAX_PAYLOAD_CHARS) -> dict[str, Any]:
        """Structured log fields describing this failure.

        The returned mapping is meant to be splatted straight into a
        ``structlog`` call so the failing endpoint and the upstream response
        body are visible in the terminal instead of being collapsed into the
        generic ``str(self)`` message.

        Keys:
            endpoint: the X path that failed (``None`` when unknown).
            status_code: the HTTP status code (``None`` when unknown).
            error: ``str(self)`` — unchanged from the previous logging shape.
            x_error: the most useful human-readable string from X's payload.
            x_payload: the JSON payload, truncated to ``max_payload_chars``.
        """
        fields: dict[str, Any] = {
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "error": str(self),
        }

        x_error = _extract_error_message(self.payload)
        if x_error is not None:
            fields["x_error"] = x_error
        else:
            fields["x_error"] = None

        fields["x_payload"] = _render_payload(self.payload, max_payload_chars)
        return fields


class RateLimitedError(XApiError):
    """HTTP 429 — the endpoint's rate limit window was exhausted."""

    def __init__(
        self,
        message: str,
        *,
        reset_at: int | None = None,
        limit: int | None = None,
        remaining: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message, status_code=429, endpoint=endpoint)
        self.reset_at = reset_at
        self.limit = limit
        self.remaining = remaining

    @property
    def retry_after_seconds(self) -> int:
        """Seconds to wait before retrying, derived from `x-rate-limit-reset`."""
        if not self.reset_at:
            return 60
        return max(1, int(self.reset_at - time.time()))


class BlockedDMError(XApiError):
    """HTTP 403 on a DM send — the recipient is unreachable (closed DMs, verified-only)."""


class AuthExpiredError(XApiError):
    """HTTP 401 — the access token is missing, expired, or revoked."""


class ForbiddenError(XApiError):
    """HTTP 403 on a non-DM endpoint (e.g. an endpoint the app has no access to)."""


class NotFoundError(XApiError):
    """HTTP 404 — the resource does not exist or is not visible to this app."""


class XCapabilityUnavailable(XApiError):
    """The endpoint exists but the current access tier does not include it.

    Raised when, for example, full-archive search (`/2/tweets/search/all`) is not
    enabled on the account.
    """


def x_error_fields(exc: BaseException) -> dict[str, Any]:
    """Structured log fields for any exception, rich for X failures.

    Use this in handlers that catch a broad ``Exception`` but may still receive
    an :class:`XApiError`. X failures get the full endpoint/status/payload
    treatment; anything else falls back to the plain ``error`` string so the
    log shape stays consistent without inventing fields that do not exist.
    """
    if isinstance(exc, XApiError):
        return exc.log_fields()
    return {"error": str(exc)}


def _extract_error_message(payload: dict[str, Any] | None) -> str | None:
    """Pull the most useful human-readable message out of an X error body."""
    if not payload:
        return None

    errors = payload.get("errors")
    if isinstance(errors, list):
        for entry in errors:
            if not isinstance(entry, dict):
                continue
            for key in _ERROR_MESSAGE_KEYS:
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    # Some X endpoints return a flat body without an ``errors`` array.
    for key in _ERROR_MESSAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _render_payload(payload: dict[str, Any] | None, max_chars: int) -> str | None:
    """Serialize an X payload for logging, truncated to ``max_chars``."""
    if not payload:
        return None

    try:
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        rendered = str(payload)

    if max_chars > 0 and len(rendered) > max_chars:
        return rendered[:max_chars] + "...(truncated)"
    return rendered
