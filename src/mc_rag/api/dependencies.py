"""FastAPI dependency providers (services, auth)."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Header

from mc_rag.application.chat_service import ChatService
from mc_rag.application.conversation_service import ConversationService
from mc_rag.application.ingestion_service import IngestionService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import AuthenticationError
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index
from mc_rag.infrastructure.llm.client import get_llm_client
from mc_rag.infrastructure.persistence.conversation_repository import ConversationRepository
from mc_rag.infrastructure.persistence.feedback_repository import FeedbackRepository
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore
from mc_rag.rag.pipeline import RagPipeline
from mc_rag.rag.retrieval import Retriever


@dataclass(slots=True, frozen=True)
class Access:
    """What the caller is allowed to see."""

    is_member: bool


def _admin_key_valid(key: str | None) -> bool:
    """Whether *key* matches the configured ``ADMIN_API_KEY`` (constant time)."""
    expected = get_settings().admin_api_key
    if not key or not expected:
        return False
    return hmac.compare_digest(key.encode(), expected.encode())


def get_access(
    authorization: Annotated[str | None, Header()] = None,
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> Access:
    """Resolve the caller's access level.

    No credentials -> public. A valid ``X-Admin-Key`` -> member (admins see
    everything). A valid ``Bearer <MEMBER_TOKEN>`` -> member. An invalid value ->
    401.
    """
    if x_admin_key is not None and _admin_key_valid(x_admin_key):
        return Access(is_member=True)

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


def require_admin(x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None) -> None:
    """FastAPI dependency: reject the request unless a valid admin key is sent."""
    if not _admin_key_valid(x_admin_key):
        raise AuthenticationError("Invalid or missing admin key.")


@lru_cache(maxsize=1)
def get_store() -> ChromaStore:
    """Process-wide vector store shared by retrieval and ingestion."""
    return ChromaStore()


@lru_cache(maxsize=1)
def get_lexical_index() -> BM25Index:
    """Process-wide BM25 index shared by retrieval, ingestion and deletion."""
    return BM25Index()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    retriever = Retriever(store=get_store(), lexical=get_lexical_index())
    pipeline = RagPipeline(retriever=retriever, llm=get_llm_client())
    conversations = ConversationService(ConversationRepository(settings.conversation_db_path))
    return ChatService(pipeline=pipeline, conversations=conversations)


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Ingestion service writing into the same collection the chat reads."""
    return IngestionService(store=get_store(), lexical=get_lexical_index())


@lru_cache(maxsize=1)
def get_feedback_repository() -> FeedbackRepository:
    return FeedbackRepository(get_settings().feedback_db_path)