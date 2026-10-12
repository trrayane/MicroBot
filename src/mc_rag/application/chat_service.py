"""Coordinates one chat request: memory in, RAG pipeline, memory out."""

from __future__ import annotations

from dataclasses import dataclass

from mc_rag.application.conversation_service import ConversationService
from mc_rag.rag.pipeline import ChatResult, RagPipeline


@dataclass(slots=True)
class ChatOutcome:
    conversation_id: str
    result: ChatResult


class ChatService:
    def __init__(self, *, pipeline: RagPipeline, conversations: ConversationService) -> None:
        self._pipeline = pipeline
        self._conversations = conversations

    def chat(
        self, message: str, *, conversation_id: str | None, is_member: bool
    ) -> ChatOutcome:
        conversation_id, history = self._conversations.open(conversation_id, is_member=is_member)
        result = self._pipeline.answer(message, is_member=is_member, history=history)
        # Only reached when the pipeline succeeded: a failed LLM call leaves no trace.
        self._conversations.add_turn(conversation_id, message, result.answer)
        return ChatOutcome(conversation_id=conversation_id, result=result)