"""OpenRouter chat-completions client (OpenAI-compatible).

Budget rules (the free tier allows 50 requests/day):

* one HTTP call per answer, at most one retry, and only for network errors/5xx;
* a 429 is **never** retried, retrying would burn more quota;
* errors become domain exceptions and never contain the API key.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import LLMRateLimitError, LLMUnavailableError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2
_DEFAULT_TIMEOUT = 45.0


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Text and token usage of one completion."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenRouterClient:
    """Minimal synchronous OpenRouter client."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
        retry_delay: float = 0.5,
        site_url: str | None = None,
        app_name: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self._timeout = timeout
        self._retry_delay = retry_delay
        self._site_url = site_url
        self._app_name = app_name
        self._transport = transport

    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """Send *messages* and return the model reply with token usage."""
        if not self._api_key:
            raise LLMUnavailableError("LLM is not configured (missing API key).")

        payload = {"model": self.model, "messages": list(messages)}
        last_reason = "unknown error"

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._post(payload)
            except httpx.TimeoutException:
                last_reason = "timeout"
            except httpx.TransportError:
                last_reason = "network error"
            else:
                status = response.status_code
                if status == 429:
                    raise LLMRateLimitError("LLM rate limit or daily quota reached.")
                if status >= 500:
                    last_reason = f"provider error {status}"
                elif status >= 400:
                    raise LLMUnavailableError(f"LLM request rejected (HTTP {status}).")
                else:
                    return self._parse(response)

            logger.warning(
                "LLM call failed (%s), attempt %d/%d", last_reason, attempt, _MAX_ATTEMPTS
            )
            if attempt < _MAX_ATTEMPTS and self._retry_delay > 0:
                time.sleep(self._retry_delay)

        raise LLMUnavailableError(f"LLM unavailable ({last_reason}).")

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(self._url, json=payload, headers=headers)

    @staticmethod
    def _parse(response: httpx.Response) -> LLMResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMUnavailableError("LLM returned a non-JSON response.") from exc
        if not isinstance(body, dict):
            raise LLMUnavailableError("LLM returned an unexpected payload.")

        error = body.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            if str(code) == "429":
                raise LLMRateLimitError("LLM rate limit or daily quota reached.")
            raise LLMUnavailableError("LLM returned an error.")

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("LLM response has no message content.") from exc
        if not isinstance(text, str):
            raise LLMUnavailableError("LLM response has no message content.")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def get_llm_client() -> OpenRouterClient:
    """Build the client from application settings."""
    settings = get_settings()
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
        site_url=settings.openrouter_site_url,
        app_name=settings.openrouter_app_name,
    )