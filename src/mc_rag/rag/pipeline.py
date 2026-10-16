"""End-to-end RAG pipeline: retrieve -> (refuse | prompt -> one LLM call) -> sources.

Cost rules: at most **one** LLM call per question (plus optional cheap extras:
follow-up suggestions and a conversation title, both disable-able), and zero
when retrieval finds nothing relevant (the refusal is predefined, in the
language of the question).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from mc_rag.infrastructure.llm.openrouter_client import LLMResponse
from mc_rag.rag.citations import Source, build_sources
from mc_rag.rag.grounding import (
    detect_language,
    language_name,
    parse_model_answer,
    refusal_message,
)
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
    followups: list[str] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


@dataclass(slots=True)
class StreamEvent:
    """One step of a streamed answer: a text delta, or the final result."""

    delta: str | None = None
    result: ChatResult | None = None


_FOLLOWUP_MAX_CHARS = 120


class RagPipeline:
    """Answers a question from the indexed corpus, honouring the caller's access."""

    def __init__(
        self,
        *,
        retriever: _Retriever,
        llm: _LLM,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
        followups_enabled: bool = False,
        followups_count: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_history = max_history_messages
        self._followups_enabled = followups_enabled
        self._followups_count = followups_count

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
            self._attach_followups(message, result, language)
        return self._finish(result, started)

    def answer_stream(
        self,
        message: str,
        *,
        is_member: bool,
        history: Sequence[dict[str, str]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream the answer: yield text deltas, then a final result event.

        Uses the LLM's ``stream`` method when available (Gemini SSE), else
        falls back to a single blocking call emitted as one delta.
        """
        started = time.perf_counter()
        language = detect_language(message)

        retrieval = self._retriever.retrieve(message, is_member=is_member, history=history)
        if not retrieval.has_answer:
            refusal = refusal_message(language)
            yield StreamEvent(delta=refusal)
            yield StreamEvent(result=self._finish(ChatResult(refusal, grounded=False), started))
            return

        messages = build_messages(
            message, retrieval.hits, history, max_history_messages=self._max_history
        )
        raw_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        stream = getattr(self._llm, "stream", None)
        if callable(stream):
            for chunk in stream(messages):  # may raise LLMRateLimitError / LLMUnavailableError
                if chunk.final:
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
                elif chunk.text:
                    raw_parts.append(chunk.text)
                    yield StreamEvent(delta=chunk.text)
        else:
            response = self._llm.complete(messages)
            raw_parts.append(response.text)
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            yield StreamEvent(delta=response.text)

        text, grounded = parse_model_answer("".join(raw_parts))
        result = ChatResult(
            answer=text,
            grounded=grounded,
            llm_calls=1 + retrieval.llm_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if not text:
            result.answer = refusal_message(language)
        elif grounded:
            result.sources = build_sources(text, retrieval.hits)
            self._attach_followups(message, result, language)
        yield StreamEvent(result=self._finish(result, started))

    def suggest_title(self, question: str) -> str | None:
        """One cheap LLM call turning the first question into a short title."""
        try:
            response = self._llm.complete(
                [
                    {
                        "role": "user",
                        "content": (
                            "Summarize this MC Club related question as a short chat "
                            "title. Use the same language as the question. Max 6 words. "
                            "Reply with the title only — no quotes, no trailing "
                            "punctuation.\n\nQuestion: " + question[:300]
                        ),
                    }
                ]
            )
        except Exception:  # noqa: BLE001 - a title is never worth failing the chat
            logger.exception("conversation title generation failed")
            return None
        title = response.text.strip().splitlines()[0].strip() if response.text.strip() else ""
        title = title.strip("\"'«» ").rstrip(".?!")[:60].strip()
        return title or None

    def _attach_followups(self, message: str, result: ChatResult, language: str) -> None:
        """Best-effort: 3 short follow-up questions from the cited sources."""
        if not self._followups_enabled or not result.sources:
            return
        titles = "; ".join(source.title for source in result.sources[:4])
        try:
            response = self._llm.complete(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Suggest exactly {self._followups_count} short follow-up "
                            "questions a student might ask next about MC Club, based on "
                            f"this conversation. Write them in {language_name(language)}. "
                            "One question per line, no numbering, no quotes, each at "
                            f"most 12 words.\n\nQuestion: {message[:300]}\n"
                            f"Source topics: {titles[:400]}"
                        ),
                    }
                ]
            )
        except Exception:  # noqa: BLE001 - suggestions are decorative
            logger.exception("follow-up suggestion failed")
            return
        result.llm_calls += 1
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        seen: set[str] = set()
        original = message.strip().casefold()
        for line in response.text.splitlines():
            question = line.strip(" -•\t\"'").rstrip(".?!").strip()
            if not (4 <= len(question) <= _FOLLOWUP_MAX_CHARS):
                continue
            key = question.casefold()
            if key in seen or key == original:
                continue
            seen.add(key)
            result.followups.append(question)
            if len(result.followups) >= self._followups_count:
                break

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