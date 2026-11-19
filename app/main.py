"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.auth_routes import router as auth_router
from app.api.compliance import router as compliance_router
from app.api.discovery import router as discovery_router
from app.api.health import router as health_router
from app.api.outreach import router as outreach_router
from app.config import get_settings
from app.db import dispose_engine
from app.errors import AppError
from app.migrations import run_migrations, wait_for_database

settings = get_settings()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("startup", app=settings.app_name, configured=settings.is_configured)

    if settings.run_migrations_on_startup:
        log.info("migrations_start")
        try:
            # The API and Postgres containers start together on ECS, so the
            # database may not be listening yet.
            await wait_for_database()
            await run_migrations()
        except Exception as exc:  # noqa: BLE001
            # A failed migration must stop the service from serving traffic
            # against a half-migrated schema.
            log.error("migrations_failed", error=str(exc))
            raise
        log.info("migrations_complete")

    try:
        # A discovery run is executed in the background, so a run that was
        # still in flight when the task stopped is picked up again here.
        from app.services.search_runner import resume_pending

        await resume_pending(settings)
        log.info("discovery_resume_checked")

        yield
    finally:
        await dispose_engine()
        log.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Find real (non-bot) accounts on X from a target segment or seed accounts, "
        "and chat with them under human approval."
    ),
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log.warning("app_error", path=request.url.path, code=exc.code, message=exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return request-validation failures in the same envelope as other errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request payload is invalid",
                "details": {"errors": exc.errors()},
            }
        },
    )


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(discovery_router)
app.include_router(outreach_router)
app.include_router(compliance_router)