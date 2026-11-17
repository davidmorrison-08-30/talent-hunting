"""Domain exceptions and their HTTP mapping."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigurationError(AppError):
    """The service is missing required configuration."""

    status_code = 503
    code = "configuration_error"


class ValidationError(AppError):
    """The request payload is invalid."""

    status_code = 422
    code = "validation_error"


class ConflictError(AppError):
    """A compliance guard rejected the operation (dedup, suppression, cap)."""

    status_code = 409
    code = "conflict"


class SuppressedError(ConflictError):
    """The recipient opted out and must not be contacted."""

    code = "suppressed"


class DuplicateOutreachError(ConflictError):
    """A live outreach record already exists for this recipient."""

    code = "duplicate_outreach"


class DailyCapError(ConflictError):
    """The application-level daily DM ceiling has been reached."""

    code = "daily_cap_reached"


class UpstreamError(AppError):
    """An upstream API (X) failed."""

    status_code = 502
    code = "upstream_error"


class NotFoundError(AppError):
    """The requested resource does not exist."""

    status_code = 404
    code = "not_found"


class RateLimitedError(AppError):
    """A local (application-side) rate limit was hit."""

    status_code = 429
    code = "rate_limited"


class LLMError(AppError):
    """The LLM provider failed or returned unusable output."""

    status_code = 502
    code = "llm_error"
