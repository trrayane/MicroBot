"""Send ONE message to the configured LLM (Gemini or OpenRouter) and print the result.

Costs 1 request of the daily quota. Usage:  python scripts/ping_llm.py
"""

from __future__ import annotations

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import LLMError, LLMRateLimitError
from mc_rag.infrastructure.llm.client import get_llm_client


def main() -> int:
    settings = get_settings()
    if settings.gemini_api_key:
        print("Provider: gemini")
        print(f"Model   : {settings.gemini_model}")
        print(f"API key : {'set' if settings.gemini_api_key else 'MISSING (check .env)'}")
    else:
        print("Provider: openrouter")
        print(f"Model   : {settings.openrouter_model}")
        print(f"API key : {'set' if settings.openrouter_api_key else 'MISSING (check .env)'}")

    try:
        response = get_llm_client().complete(
            [{"role": "user", "content": "Reply with the single word: OK"}]
        )
    except LLMRateLimitError:
        print("FAILED: rate limit or daily quota reached (429).")
        return 1
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"Answer  : {response.text!r}")
    print(f"Tokens  : prompt={response.prompt_tokens} completion={response.completion_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
