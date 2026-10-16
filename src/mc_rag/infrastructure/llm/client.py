"""LLM client selection: native Gemini when a Gemini key is configured, else OpenRouter."""

from __future__ import annotations

import logging

from mc_rag.config import get_settings
from mc_rag.infrastructure.llm.gemini_client import GeminiClient
from mc_rag.infrastructure.llm.openrouter_client import OpenRouterClient, get_openrouter_client

logger = logging.getLogger(__name__)


def _masked_key(key: str | None) -> str:
    if not key:
        return "unset"
    return f"{key[:6]}****" if len(key) >= 6 else "****"


def log_llm_configuration() -> None:
    """Log the selected provider and masked key identity at process startup."""
    settings = get_settings()
    if settings.gemini_api_key:
        logger.info(
            "LLM provider=Google Gemini key_variable=GEMINI_API_KEY "
            "(GOOGLE_API_KEY alias supported) key=%s model=%s",
            _masked_key(settings.gemini_api_key),
            settings.gemini_model,
        )
    elif settings.openrouter_api_key:
        logger.info(
            "LLM provider=OpenRouter key_variable=OPENROUTER_API_KEY key=%s model=%s",
            _masked_key(settings.openrouter_api_key),
            settings.openrouter_model,
        )
    else:
        logger.warning("LLM provider is unconfigured; no supported API key is set.")


def get_llm_client() -> GeminiClient | OpenRouterClient:
    """Return the LLM client matching the configured API keys.

    A ``GEMINI_API_KEY`` takes precedence; otherwise fall back to OpenRouter.
    """
    settings = get_settings()
    if settings.gemini_api_key:
        return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    return get_openrouter_client()