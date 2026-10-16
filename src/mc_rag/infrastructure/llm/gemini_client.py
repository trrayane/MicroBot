"""Native Google Generative Language (Gemini) LLM client.

Talks directly to ``generativelanguage.googleapis.com`` with a Google AI API key
(``AIza...``). Implements the same :meth:`complete` surface as the OpenRouter
client so :class:`mc_rag.rag.pipeline.RagPipeline` is untouched.

When several keys are configured (``GEMINI_API_KEYS`` comma-separated list,
plus the single ``GEMINI_API_KEY``), the client runs a **key roulette**: a key
that returns 429 (quota exhausted) is put on cooldown and the request is
retried on the next key of the pool until one succeeds or every key is
cooling down.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from mc_rag.domain.exceptions import LLMRateLimitError, LLMUnavailableError
from mc_rag.infrastructure.llm.api_key_pool import ApiKeyPool
from mc_rag.infrastructure.llm.openrouter_client import LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TIMEOUT = 45.0
_DEFAULT_COOLDOWN_SECONDS = 600.0
_RETRY_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s")


class GeminiClient:
    """Minimal synchronous client for the ``models/{model}:generateContent`` API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_keys: Sequence[str] | None = None,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        keys = [key for key in (api_keys or []) if key]
        if api_key:
            keys.insert(0, api_key)
        self._pool = ApiKeyPool(keys, cooldown_seconds=cooldown_seconds)
        self.model = model
        self._timeout = timeout
        self._transport = transport

    @property
    def pool(self) -> ApiKeyPool:
        """The key roulette backing this client."""
        return self._pool

    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """Send *messages* to Gemini and return the reply with token usage.

        On a 429 the current key is cooled down (honouring the API's
        ``retryDelay`` when present) and the call is retried on the next
        key of the pool.
        """
        if self._pool.size == 0:
            raise LLMUnavailableError("LLM is not configured (missing Gemini API key).")

        last_rate_limit: LLMRateLimitError | None = None
        for _ in range(self._pool.size):
            key = self._pool.get()
            if key is None:
                break
            try:
                response = self._post(messages, key)
                result = self._parse(response, key)
                self._pool.record_usage(
                    key,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                return result
            except LLMRateLimitError as exc:
                last_rate_limit = exc
                self._pool.penalize(key, getattr(exc, "retry_delay", None))
                logger.warning(
                    "Gemini key %s hit its quota (%s) — cooled down %.0fs, %d key(s) left",
                    f"{key[:6]}****" if len(key) >= 6 else "****",
                    exc,
                    getattr(exc, "retry_delay", None) or self._pool.cooldown_seconds,
                    max(self._pool.available(), 0),
                )
        if last_rate_limit is not None:
            raise last_rate_limit
        raise LLMRateLimitError("All Gemini API keys are rate-limited; retry later.")

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[LLMStreamChunk]:
        """Stream the reply via ``streamGenerateContent`` (SSE).

        Yields text deltas as they arrive, then a final chunk carrying the
        token usage. The key roulette applies before the first byte: a 429
        while opening the stream cools the key down and retries on the next.
        Once bytes have been yielded a mid-stream failure is raised as-is
        (the caller already received part of the answer).
        """
        if self._pool.size == 0:
            raise LLMUnavailableError("LLM is not configured (missing Gemini API key).")

        last_rate_limit: LLMRateLimitError | None = None
        for _ in range(self._pool.size):
            key = self._pool.get()
            if key is None:
                break
            emitted = False
            try:
                for chunk in self._stream_once(messages, key):
                    emitted = True
                    if chunk.final:
                        self._pool.record_usage(
                            key,
                            prompt_tokens=chunk.prompt_tokens,
                            completion_tokens=chunk.completion_tokens,
                        )
                    yield chunk
                return
            except LLMRateLimitError as exc:
                if emitted:
                    raise
                last_rate_limit = exc
                self._pool.penalize(key, getattr(exc, "retry_delay", None))
                logger.warning(
                    "Gemini key %s hit its quota while streaming (%s) — cooled down %.0fs, "
                    "%d key(s) left",
                    f"{key[:6]}****" if len(key) >= 6 else "****",
                    exc,
                    getattr(exc, "retry_delay", None) or self._pool.cooldown_seconds,
                    max(self._pool.available(), 0),
                )
        if last_rate_limit is not None:
            raise last_rate_limit
        raise LLMRateLimitError("All Gemini API keys are rate-limited; retry later.")

    def _stream_once(
        self, messages: Sequence[dict[str, str]], api_key: str
    ) -> Iterator[LLMStreamChunk]:
        """Open one SSE stream on a single key and yield its chunks."""
        payload = self._payload(messages)
        url = f"{_BASE_URL}/models/{self.model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        body = self._redact(body, api_key)
                        logger.warning(
                            "Gemini API stream error status=%d response_body=%s",
                            response.status_code,
                            body,
                        )
                        error: Any = None
                        try:
                            parsed = json.loads(body)
                            error = parsed.get("error") if isinstance(parsed, dict) else None
                        except ValueError:
                            pass
                        message = (
                            str(error.get("message", f"HTTP {response.status_code}"))
                            if isinstance(error, dict)
                            else f"HTTP {response.status_code}"
                        )
                        if response.status_code == 429:
                            exc = LLMRateLimitError(f"Gemini API rate limit/quota (429): {message}")
                            exc.retry_delay = self._retry_delay(error)
                            raise exc
                        raise LLMUnavailableError(f"LLM request rejected ({message}).")

                    prompt_tokens = 0
                    completion_tokens = 0
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        try:
                            frame = json.loads(data)
                        except ValueError:
                            continue
                        if not isinstance(frame, dict):
                            continue
                        usage = frame.get("usageMetadata")
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("promptTokenCount") or 0)
                            completion_tokens = int(usage.get("candidatesTokenCount") or 0)
                        text = self._frame_text(frame)
                        if text:
                            yield LLMStreamChunk(text=text)
                    yield LLMStreamChunk(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        final=True,
                    )
        except httpx.TimeoutException:
            raise LLMUnavailableError("LLM unavailable (timeout).") from None
        except httpx.TransportError:
            raise LLMUnavailableError("LLM unavailable (network error).") from None

    @staticmethod
    def _frame_text(frame: dict[str, Any]) -> str:
        """Extract the text delta of one SSE frame ("" when absent)."""
        try:
            text = frame["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return ""
        return text if isinstance(text, str) else ""

    def _redact(self, body: str, api_key: str | None = None) -> str:
        """Replace every known key appearing in *body* with ``[REDACTED]``."""
        secrets = {key for key in [api_key] if key} | set(self._pool.raw_keys())
        for secret in secrets:
            body = body.replace(secret, "[REDACTED]")
        return body

    def _post(self, messages: Sequence[dict[str, str]], api_key: str) -> httpx.Response:
        payload = self._payload(messages)
        url = f"{_BASE_URL}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        kwargs: dict[str, Any] = {}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        try:
            return httpx.post(url, json=payload, headers=headers, timeout=self._timeout, **kwargs)
        except httpx.TimeoutException:
            raise LLMUnavailableError("LLM unavailable (timeout).") from None
        except httpx.TransportError:
            raise LLMUnavailableError("LLM unavailable (network error).") from None

    @staticmethod
    def _retry_delay(error: object) -> float | None:
        """Extract the API's ``retryDelay`` (seconds) from a 429 error body."""
        if not isinstance(error, dict):
            return None
        details = error.get("details")
        candidates: list[object] = list(details) if isinstance(details, list) else [details]
        candidates.append(error)
        for item in candidates:
            if not isinstance(item, dict):
                continue
            raw = item.get("retryDelay")
            if raw is None:
                continue
            match = _RETRY_DELAY_RE.search(str(raw))
            if match:
                return max(float(match.group(1)), 1.0)
        return None

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

    def _parse(self, response: httpx.Response, api_key: str | None = None) -> LLMResponse:
        response_body = self._redact(response.text, api_key)
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
                exc = LLMRateLimitError(f"Gemini API rate limit/quota (429): {message}")
                exc.retry_delay = self._retry_delay(error)
                raise exc
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