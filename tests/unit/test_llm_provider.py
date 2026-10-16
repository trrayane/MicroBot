from __future__ import annotations

import logging

import httpx
import pytest

from mc_rag.config import Settings
from mc_rag.domain.exceptions import LLMRateLimitError
from mc_rag.infrastructure.llm import client as llm_client
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