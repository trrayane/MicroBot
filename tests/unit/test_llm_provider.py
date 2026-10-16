from __future__ import annotations

import logging

import httpx
import pytest

from mc_rag.config import Settings
from mc_rag.domain.exceptions import LLMRateLimitError
from mc_rag.infrastructure.llm import client as llm_client
from mc_rag.infrastructure.llm import gemini_client as gemini_module
from mc_rag.infrastructure.llm.api_key_pool import ApiKeyPool
from mc_rag.infrastructure.llm.gemini_client import GeminiClient


def test_google_api_key_alias_selects_gemini_and_logs_masked_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "AIzaSyRegressionTestKey"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    with caplog.at_level(logging.INFO):
        selected = llm_client.get_llm_client()
        llm_client.log_llm_configuration()

    assert isinstance(selected, GeminiClient)
    assert settings.gemini_api_key == secret
    assert "provider=Google Gemini" in caplog.text
    assert "key_variable=GEMINI_API_KEY" in caplog.text
    assert "AIzaSy****" in caplog.text
    assert secret not in caplog.text


def test_gemini_429_logs_full_quota_details_without_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "AIzaSyRegressionTestKey"
    response = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Requests per minute quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"retryDelay": "60s"}],
            }
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/test"),
    )
    client = GeminiClient(api_key=secret, model="test-model")

    with caplog.at_level(logging.WARNING), pytest.raises(
        LLMRateLimitError, match="Requests per minute quota exceeded"
    ):
        client._parse(response)

    assert "RESOURCE_EXHAUSTED" in caplog.text
    assert "retryDelay" in caplog.text
    assert secret not in caplog.text


# --------------------------------------------------------------------------- #
# Gemini key roulette (GEMINI_API_KEYS pool, 429 → next key)
# --------------------------------------------------------------------------- #

_OK_BODY = {
    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
}


def _quota_response() -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"retryDelay": "30s"}],
            }
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/test"),
    )


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json=_OK_BODY,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/test"),
    )


def test_key_pool_round_robin_and_penalize() -> None:
    pool = ApiKeyPool(["k1", "k2", "k3"], cooldown_seconds=60.0)
    assert [pool.get() for _ in range(3)] == ["k1", "k2", "k3"]
    pool.penalize("k2", 60.0)
    assert [pool.get() for _ in range(3)] == ["k1", "k3", "k1"]
    pool.reset("k2")
    assert pool.get() == "k2"


def test_key_pool_skips_cooled_keys_and_reports_empty() -> None:
    pool = ApiKeyPool(["k1", "k2"], cooldown_seconds=60.0)
    pool.penalize("k1", 60.0)
    pool.penalize("k2", 60.0)
    assert pool.get() is None
    pool.reset()
    assert pool.get() == "k1"


def test_roulette_uses_next_key_after_429(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_post(url: str, *, json: dict, headers: dict, timeout: float) -> httpx.Response:
        key = headers["x-goog-api-key"]
        seen.append(key)
        return _quota_response() if key == "key-one" else _ok_response()

    monkeypatch.setattr(gemini_module.httpx, "post", fake_post)
    client = GeminiClient(api_keys=["key-one", "key-two"], model="test-model")

    reply = client.complete([{"role": "user", "content": "salut"}])

    assert reply.text == "ok"
    assert seen == ["key-one", "key-two"]


def test_roulette_cools_key_using_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_quota(url: str, *, json: dict, headers: dict, timeout: float) -> httpx.Response:
        return _quota_response()

    monkeypatch.setattr(gemini_module.httpx, "post", always_quota)
    client = GeminiClient(
        api_keys=["key-one", "key-two"], model="test-model", cooldown_seconds=600.0
    )

    with pytest.raises(LLMRateLimitError, match="Quota exceeded"):
        client.complete([{"role": "user", "content": "salut"}])

    import time as _time

    # Both keys cooled down (retryDelay=30s from the error body).
    assert client.pool.available() == 0
    assert client.pool.get() is None
    # Cooldown honours the API-provided retryDelay (~30s, not 600s).
    for blocked_until in client.pool._blocked_until.values():  # noqa: SLF001
        remaining = blocked_until - _time.monotonic()
        assert 0 < remaining <= 31


def test_roulette_single_key_429_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_quota(url: str, *, json: dict, headers: dict, timeout: float) -> httpx.Response:
        return _quota_response()

    monkeypatch.setattr(gemini_module.httpx, "post", always_quota)
    client = GeminiClient(api_keys=["only-key"], model="test-model")

    with pytest.raises(LLMRateLimitError):
        client.complete([{"role": "user", "content": "salut"}])


def test_settings_merge_single_key_into_roulette_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEYS", "k-a, k-b,,k-a")
    monkeypatch.setenv("GEMINI_API_KEY", "k-single")
    settings = Settings(_env_file=None)
    assert settings.gemini_key_list == ["k-a", "k-b", "k-single"]


def test_factory_selects_gemini_with_roulette_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEYS", "k-a,k-b")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    selected = llm_client.get_llm_client()

    assert isinstance(selected, GeminiClient)
    assert selected.pool.size == 2