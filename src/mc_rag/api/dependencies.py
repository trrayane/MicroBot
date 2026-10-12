"""FastAPI dependency providers (services, auth)."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Header

from mc_rag.application.chat_service import ChatService
from mc_rag.application.conversation_service import ConversationService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import AuthenticationError
from mc_rag.infrastructure.llm.openrouter_client import get_llm_client
from mc_rag.infrastructure.persistence.conversation_repository import ConversationRepository
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore
from mc_rag.rag.pipeline import RagPipeline
from mc_rag.rag.retrieval import Retriever


@dataclass(slots=True, frozen=True)
class Access:
    """What the caller is allowed to see."""

    is_member: bool


def get_access(authorization: Annotated[str | None, Header()] = None) -> Access:
    """No header -> public. Valid ``Bearer <MEMBER_TOKEN>`` -> member. Anything else -> 401."""
    if authorization is None:
        return Access(is_member=False)

    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    expected = get_settings().member_token
    valid = (
        scheme.lower() == "bearer"
        and bool(token)
        and bool(expected)
        and hmac.compare_digest(token.encode(), (expected or "").encode())
    )
    if not valid:
        raise AuthenticationError("Invalid or missing member token.")
    return Access(is_member=True)


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    retriever = Retriever(store=ChromaStore())
    pipeline = RagPipeline(retriever=retriever, llm=get_llm_client())
    conversations = ConversationService(ConversationRepository(settings.conversation_db_path))
    return ChatService(pipeline=pipeline, conversations=conversations)