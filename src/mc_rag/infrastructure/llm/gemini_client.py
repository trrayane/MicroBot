"""Native Google Generative Language (Gemini) LLM client.

Talks directly to ``generativelanguage.googleapis.com`` with a Google AI API key
(``AIza...``). Implements the same :meth:`complete` surface as the OpenRouter
client so :class:`mc_rag.rag.pipeline.RagPipeline` is untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from mc_rag.domain.exceptions import LLMRateLimitError, LLMUnavailableError
from mc_rag.infrastructure.llm.openrouter_client import LLMResponse

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TIMEOUT = 45.0


class GeminiClient:
    """Minimal synchronous client for the ``models/{model}:generateContent`` API."""

    def __init__(
        self, *, api_key: str | None, model: str, timeout: float = _DEFAULT_TIMEOUT
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._timeout = timeout

    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """Send *messages* to Gemini and return the reply with token usage."""
        if not self._api_key:
            raise LLMUnavailableError("LLM is not configured (missing Gemini API key).")

        payload = self._payload(messages)
        url = f"{_BASE_URL}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException:
            raise LLMUnavailableError("LLM unavailable (timeout).") from None
        except httpx.TransportError:
            raise LLMUnavailableError("LLM unavailable (network error).") from None
        return self._parse(response)

    @staticmethod
    def _payload(messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        """Split the OpenAI-style turns into ``system_instruction`` + ``contents``.

        Gemini only accepts ``user``/``model`` content roles and requires no
        empty turns, so assistant turns map to ``model`` and empties are dropped.
        """
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in ("system", "developer"):
                if content.strip():
                    system_parts.append({"text": content})
                continue
            if not content.strip() and contents:
                continue
            contents.append(
                {"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]}
            )

        # Collapse consecutive same-role turns (the API rejects them).
        deduped: list[dict[str, Any]] = []
        for item in contents:
            if deduped and deduped[-1]["role"] == item["role"]:
                deduped.pop()
            deduped.append(item)

        payload: dict[str, Any] = {"contents": deduped}
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}
        return payload

    def _parse(self, response: httpx.Response) -> LLMResponse:
        response_body = response.text
        if self._api_key:
            response_body = response_body.replace(self._api_key, "[REDACTED]")
        try:
            body = response.json()
        except ValueError as exc:
            logger.warning(
                "Gemini API error status=%d response_body=%s",
                response.status_code,
                response_body,
            )
            raise LLMUnavailableError("LLM returned a non-JSON response.") from exc
        if not isinstance(body, dict):
            logger.warning(
                "Gemini API error status=%d response_body=%s",
                response.status_code,
                response_body,
            )
            raise LLMUnavailableError("LLM returned an unexpected payload.")

        if response.status_code >= 400 or "error" in body:
            logger.warning(
                "Gemini API error status=%d response_body=%s",
                response.status_code,
                response_body,
            )
            error = body.get("error")
            message = (
                str(error.get("message", f"HTTP {response.status_code}"))
                if isinstance(error, dict)
                else f"HTTP {response.status_code}"
            )
            if response.status_code == 429:
                raise LLMRateLimitError(f"Gemini API rate limit/quota (429): {message}")
            raise LLMUnavailableError(f"LLM request rejected ({message}).")

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("LLM response has no message content.") from exc
        if not isinstance(text, str):
            raise LLMUnavailableError("LLM response has no message content.")

        usage = body.get("usageMetadata")
        usage = usage if isinstance(usage, dict) else {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
        )