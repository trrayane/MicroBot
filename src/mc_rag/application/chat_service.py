"""Coordinates one chat request: memory in, RAG pipeline, memory out."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from mc_rag.application.conversation_service import ConversationService
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository
from mc_rag.rag.pipeline import ChatResult, RagPipeline, StreamEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatOutcome:
    conversation_id: str
    query_log_id: int
    result: ChatResult
    title: str | None = None


@dataclass(slots=True)
class StreamChatEvent:
    """One SSE-ready event of a streamed chat turn."""

    type: str  # "delta" | "done"
    delta: str | None = None
    payload: dict | None = None


class ChatService:
    def __init__(
        self,
        *,
        pipeline: RagPipeline,
        conversations: ConversationService,
        query_logs: QueryLogRepository,
    ) -> None:
        self._pipeline = pipeline
        self._conversations = conversations
        self._query_logs = query_logs

    def chat(
        self, message: str, *, conversation_id: str | None, is_member: bool
    ) -> ChatOutcome:
        conversation_id, history, title = self._conversations.open(
            conversation_id, is_member=is_member
        )
        result = self._pipeline.answer(message, is_member=is_member, history=history)
        # Only reached when the pipeline succeeded: a failed LLM call leaves no trace.
        query_log_id = self._query_logs.add_query(conversation_id, message)
        self._conversations.add_turn(conversation_id, message, result.answer)
        title = self._maybe_title(conversation_id, message, title)
        return ChatOutcome(
            conversation_id=conversation_id,
            query_log_id=query_log_id,
            result=result,
            title=title,
        )

    def chat_stream(
        self, message: str, *, conversation_id: str | None, is_member: bool
    ) -> Iterator[StreamChatEvent]:
        """Stream a chat turn: delta events, then one done event with the payload."""
        conversation_id, history, title = self._conversations.open(
            conversation_id, is_member=is_member
        )
        result: ChatResult | None = None
        for event in self._pipeline.answer_stream(
            message, is_member=is_member, history=history
        ):
            if event.delta is not None:
                yield StreamChatEvent(type="delta", delta=event.delta)
            elif event.result is not None:
                result = event.result
        if result is None:  # pragma: no cover - the pipeline always yields a result
            return
        query_log_id = self._query_logs.add_query(conversation_id, message)
        self._conversations.add_turn(conversation_id, message, result.answer)
        title = self._maybe_title(conversation_id, message, title)
        yield StreamChatEvent(
            type="done",
            payload={
                "conversation_id": conversation_id,
                "query_log_id": query_log_id,
                "answer": result.answer,
                "grounded": result.grounded,
                "title": title,
                "followups": result.followups,
                "sources": [
                    {"doc_id": s.doc_id, "title": s.title, "snippet": s.snippet}
                    for s in result.sources
                ],
                "usage": {
                    "llm_calls": result.llm_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
                "latency_ms": result.latency_ms,
            },
        )

    def _maybe_title(
        self, conversation_id: str, message: str, existing: str | None
    ) -> str | None:
        """Generate and persist a title on the first turn of a conversation."""
        if existing:
            return existing
        title = self._pipeline.suggest_title(message)
        if title:
            try:
                self._conversations.set_title(conversation_id, title)
            except Exception:  # noqa: BLE001 - a title must never fail the chat
                logger.exception("failed to persist conversation title")
        return title
