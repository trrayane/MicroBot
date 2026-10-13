"""LLM client selection: native Gemini when a Gemini key is configured, else OpenRouter."""

from __future__ import annotations

from mc_rag.config import get_settings
from mc_rag.infrastructure.llm.gemini_client import GeminiClient
from mc_rag.infrastructure.llm.openrouter_client import OpenRouterClient, get_openrouter_client


def get_llm_client() -> GeminiClient | OpenRouterClient:
    """Return the LLM client matching the configured API keys.

    A ``GEMINI_API_KEY`` takes precedence; otherwise fall back to OpenRouter.
    """
    settings = get_settings()
    if settings.gemini_api_key:
        return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    return get_openrouter_client()