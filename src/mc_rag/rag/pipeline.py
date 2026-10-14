"""End-to-end RAG pipeline: retrieve -> (refuse | prompt -> one LLM call) -> sources.

Cost rules: at most **one** LLM call per question, and zero when retrieval finds
nothing relevant (the refusal is predefined, in the language of the question).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from mc_rag.infrastructure.llm.openrouter_client import LLMResponse
from mc_rag.rag.citations import Source, build_sources
from mc_rag.rag.grounding import detect_language, parse_model_answer, refusal_message
from mc_rag.rag.prompts import DEFAULT_MAX_HISTORY_MESSAGES, build_messages
from mc_rag.rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def retrieve(
        self,
        message: str,
        *,
        is_member: bool,
        history: Sequence[dict[str, str]] | None = None,
    ) -> RetrievalResult: ...


class _LLM(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse: ...


@dataclass(slots=True)
class ChatResult:
    """What the API needs to build its response."""

    answer: str
    grounded: bool
    sources: list[Source] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class RagPipeline:
    """Answers a question from the indexed corpus, honouring the caller's access."""

    def __init__(
        self,
        *,
        retriever: _Retriever,
        llm: _LLM,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_history = max_history_messages

    def answer(
        self,
        message: str,
        *,
        is_member: bool,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        language = detect_language(message)

        retrieval = self._retriever.retrieve(message, is_member=is_member, history=history)
        if not retrieval.has_answer:
            return self._finish(ChatResult(refusal_message(language), grounded=False), started)

        messages = build_messages(
            message, retrieval.hits, history, max_history_messages=self._max_history
        )
        response = self._llm.complete(messages)  # may raise LLMRateLimitError / LLMUnavailableError

        text, grounded = parse_model_answer(response.text)
        result = ChatResult(
            answer=text,
            grounded=grounded,
            llm_calls=1 + retrieval.llm_calls,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        if not text:
            result.answer = refusal_message(language)
        elif grounded:
            result.sources = build_sources(text, retrieval.hits)
        return self._finish(result, started)

    @staticmethod
    def _finish(result: ChatResult, started: float) -> ChatResult:
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "chat grounded=%s llm_calls=%d prompt_tokens=%d completion_tokens=%d latency_ms=%d",
            result.grounded,
            result.llm_calls,
            result.prompt_tokens,
            result.completion_tokens,
            result.latency_ms,
        )
        return result