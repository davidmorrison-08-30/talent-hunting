"""AWS Bedrock runtime wrapper for structured LLM output.

Per `build-docs/how-to-use-llm.md`, inference uses the AWS Bedrock runtime via
``boto3``. The model id, region and bearer token are supplied through
environment variables.

Every agent stage needs machine-readable output, so this wrapper uses Bedrock's
``converse`` API with a single JSON-schema tool and returns the parsed object.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import boto3
import structlog

from app.config import Settings, get_settings
from app.errors import ConfigurationError, LLMError

log = structlog.get_logger(__name__)


class BedrockLLM:
    """Async-friendly wrapper over the Bedrock ``converse`` API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._semaphore = asyncio.Semaphore(max(1, self.settings.llm_max_concurrency))

    # -- client ------------------------------------------------------------
    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.settings.aws_bedrock_model_id:
            raise ConfigurationError(
                "AWS_BEDROCK_MODEL_ID must be set to run LLM inference"
            )

        kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        if self.settings.aws_default_profile:
            kwargs["profile_name"] = self.settings.aws_default_profile
        if self.settings.aws_bearer_token_bedrock:
            # botocore reads a Bedrock API key from AWS_BEARER_TOKEN_BEDROCK and
            # then prefers "bearer" signature version over SigV4. Passing the key
            # as aws_session_token would instead break the credential chain.
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = (
                self.settings.aws_bearer_token_bedrock
            )
            # An empty AWS_DEFAULT_PROFILE makes botocore try to resolve a
            # profile named "" and fail, so it is cleared when unset.
            if not self.settings.aws_default_profile:
                os.environ.pop("AWS_DEFAULT_PROFILE", None)
                os.environ.pop("AWS_PROFILE", None)

        try:
            self._client = boto3.client("bedrock-runtime", **kwargs)
        except Exception as exc:  # pragma: no cover - depends on AWS environment
            raise ConfigurationError(
                f"Could not create a Bedrock runtime client: {exc}"
            ) from exc
        return self._client

    # -- inference ---------------------------------------------------------
    async def converse_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Run one inference call and return the structured JSON object.

        ``schema`` is a JSON Schema object; Bedrock is instructed to call a
        single tool whose input conforms to it, and the tool input is returned.
        """
        client = self._build_client()

        tool_spec: dict[str, Any] = {
            "toolSpec": {
                "name": "structured_output",
                "description": "Return the requested structured result.",
                "inputSchema": {"json": schema},
            }
        }

        request: dict[str, Any] = {
            "modelId": self.settings.aws_bedrock_model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "system": [{"text": system}],
            "toolConfig": {
                "tools": [tool_spec],
                "toolChoice": {"tool": {"name": "structured_output"}},
            },
            "inferenceConfig": {
                "maxTokens": max_tokens or self.settings.llm_max_tokens,
                "temperature": temperature,
            },
        }

        async with self._semaphore:
            response = await self._converse(client, request)

        return _extract_tool_input(response)

    async def _converse(self, client: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Call Bedrock, running the blocking SDK call in a worker thread.

        The call is bounded by ``llm_timeout_seconds``. Without a bound a
        stalled Bedrock request would hold the HTTP request open until the
        API Gateway gives up on it, and the caller would never learn what
        happened.
        """
        last_error: Exception | None = None
        timeout = self.settings.llm_timeout_seconds

        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(client.converse, **request),
                    timeout=timeout,
                )
            except TimeoutError:
                last_error = LLMError(
                    "AWS Bedrock inference timed out",
                    details={"timeout_seconds": timeout},
                )
                log.warning("bedrock_timeout", attempt=attempt, timeout_seconds=timeout)
                break
            except Exception as exc:
                last_error = exc
                if _is_throttling(exc) and attempt == 0:
                    log.warning("bedrock_throttled", attempt=attempt)
                    await asyncio.sleep(2)
                    continue
                break

        raise LLMError(
            "AWS Bedrock inference failed",
            details={"error": str(last_error)},
        )

    async def health(self) -> dict[str, Any]:
        """Lightweight check that the client can be constructed."""
        try:
            self._build_client()
        except ConfigurationError as exc:
            return {"status": "error", "detail": exc.message}
        return {
            "status": "ok",
            "model_id": self.settings.aws_bedrock_model_id,
            "region": self.settings.aws_region,
        }


def _is_throttling(exc: Exception) -> bool:
    text = str(exc).lower()
    return "throttling" in text or "too many requests" in text


def _extract_tool_input(response: dict[str, Any]) -> dict[str, Any]:
    """Pull the tool input out of a Bedrock converse response."""
    output = response.get("output") or {}
    message = output.get("message") or {}
    content = message.get("content") or []

    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if not tool_use:
            continue
        tool_input = tool_use.get("input")
        if isinstance(tool_input, dict):
            return tool_input
        if isinstance(tool_input, str):
            try:
                parsed = json.loads(tool_input)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    "Bedrock returned a tool input that is not valid JSON",
                    details={"error": str(exc), "raw": tool_input[:500]},
                ) from exc
            if isinstance(parsed, dict):
                return parsed

    text = " ".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    ).strip()
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Bedrock returned free text where structured output was expected",
                details={"error": str(exc), "raw": text[:500]},
            ) from exc
        if isinstance(parsed, dict):
            return parsed

    raise LLMError(
        "Bedrock returned no usable structured output",
        details={"response_keys": sorted(response.keys())},
    )


#: Process-wide client shared by the agents.
_bedrock: BedrockLLM | None = None


def get_llm() -> BedrockLLM:
    """Return the shared Bedrock client."""
    global _bedrock
    if _bedrock is None:
        _bedrock = BedrockLLM()
    return _bedrock
