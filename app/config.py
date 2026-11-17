"""Application configuration loaded from environment variables.

All secrets are read from the environment only; nothing is hardcoded. See `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------
    app_name: str = "X Talent-Hunting APIs"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Database ------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/talent_hunting",
        description="Async SQLAlchemy URL for PostgreSQL.",
    )
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- X API (OAuth 2.0 user context) ---------------------------------
    x_client_id: str = Field(default="", description="OAuth 2.0 consumer key of the X app.")
    x_client_secret: str = Field(default="", description="OAuth 2.0 consumer secret of the X app.")
    x_redirect_uri: str = Field(
        default="http://localhost:8000/auth/x/callback",
        description="Registered OAuth 2.0 redirect URI.",
    )
    x_bearer_token: str = Field(
        default="",
        description="Optional app-only bearer token used for read-only endpoints if available.",
    )
    x_api_base_url: str = "https://api.x.com"
    x_oauth_base_url: str = "https://x.com/i/oauth2"
    x_token_url: str = "https://api.x.com/2/oauth2/token"
    x_scopes: str = "tweet.read users.read dm.read dm.write offline.access"
    x_timeout_seconds: float = 30.0
    x_max_retries: int = 3
    x_retry_max_wait_seconds: float = Field(
        default=900.0,
        gt=0,
        description=(
            "Upper bound on how long a 429 retry waits before giving up. "
            "Lower it in tests to keep the suite fast."
        ),
    )

    # --- AWS Bedrock ----------------------------------------------------
    aws_bedrock_model_id: str = Field(
        default="",
        description="Bedrock model id, e.g. anthropic.claude-3-5-sonnet-20241022-v2:0.",
    )
    aws_region: str = Field(default="us-east-1")
    aws_bearer_token_bedrock: str = Field(
        default="", description="Optional Bedrock bearer token."
    )
    aws_default_profile: str = ""
    llm_timeout_seconds: float = 60.0
    llm_max_concurrency: int = 4
    llm_max_tokens: int = 2048

    # --- Pacing / compliance --------------------------------------------
    dm_daily_cap: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Application-level daily DM ceiling (X hard cap is 500/day).",
    )
    dm_min_delay_seconds: float = 30.0
    dm_max_delay_seconds: float = 90.0
    dm_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Ratio above which a draft is considered near-duplicate of a prior send.",
    )
    dm_poll_interval_seconds: int = 240
    discovery_spend_ceiling_usd: float = Field(
        default=5.0,
        description="Maximum estimated pay-per-use spend for a single discovery run.",
    )
    post_read_cost_usd: float = 0.005
    user_read_cost_usd: float = 0.010

    # --- Runtime knobs ---------------------------------------------------
    run_migrations_on_startup: bool = True
    enable_dispatcher: bool = Field(
        default=True,
        description="Run the in-process drip dispatcher for queued DM approvals.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        """True when the minimum credentials for X OAuth and Bedrock are present."""
        return bool(self.x_client_id and self.x_client_secret and self.aws_bedrock_model_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
