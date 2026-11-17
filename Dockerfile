# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Talent Hunting backend image.
#
# Multi-stage: the builder resolves the uv.lock into a virtualenv, the runtime
# stage carries only that virtualenv and the application source.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer: resolve from the lock file only, so this layer is cached
# until pyproject.toml / uv.lock actually change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Application layer.
COPY alembic ./alembic
COPY alembic.ini ./
COPY app ./app
COPY scripts ./scripts
RUN uv sync --frozen --no-dev


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000

# Liveness probe target used by the ECS target group as well.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Exec form (not shell form) so SIGTERM reaches uvicorn and the FastAPI
# lifespan shutdown hook runs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
