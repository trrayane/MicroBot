"""FastAPI dependency providers (services, auth)."""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mc_rag.application.authentication_service import AuthenticationService
from mc_rag.application.chat_service import ChatService
from mc_rag.application.conversation_service import ConversationService
from mc_rag.application.ingestion_service import IngestionService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import AuthenticationError
from mc_rag.infrastructure.discord.membership_verifier import DiscordMembershipVerifier
from mc_rag.infrastructure.discord.oauth_client import DiscordOAuthClient
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index
from mc_rag.infrastructure.llm.client import get_llm_client
from mc_rag.infrastructure.persistence.conversation_repository import ConversationRepository
from mc_rag.infrastructure.persistence.feedback_repository import FeedbackRepository
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository
from mc_rag.infrastructure.persistence.user_store import UserStore
from mc_rag.infrastructure.tokens import JwtCodec
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore
from mc_rag.rag.pipeline import RagPipeline
from mc_rag.rag.retrieval import Retriever

_bearer = HTTPBearer(auto_error=False)


@dataclass(slots=True, frozen=True)
class Access:
    """What the caller is allowed to see."""

    is_member: bool
    is_admin: bool = False
    user_id: str | None = None
    display_name: str | None = None
    avatar: str | None = None


def _admin_key_valid(key: str | None) -> bool:
    """Whether *key* matches the configured ``ADMIN_API_KEY`` (constant time)."""
    expected = get_settings().admin_api_key
    if not key or not expected:
        return False
    return hmac.compare_digest(key.encode(), expected.encode())


@lru_cache(maxsize=1)
def get_user_store() -> UserStore:
    """Process-wide SQLite store for Discord users and document ownership."""
    return UserStore(get_settings().users_db_path)


@lru_cache(maxsize=1)
def get_jwt_codec() -> JwtCodec:
    settings = get_settings()
    return JwtCodec(settings.jwt_secret_key, settings.jwt_access_token_expire_minutes)


@lru_cache(maxsize=1)
def get_authentication_service() -> AuthenticationService:
    """Build the Discord authentication service (OAuth + JWT + user store)."""
    import httpx

    settings = get_settings()
    if not (settings.discord_client_id and settings.discord_client_secret
            and settings.discord_guild_id):
        raise AuthenticationError("Discord OAuth is not configured.")

    http = httpx.AsyncClient(timeout=8.0)
    oauth = DiscordOAuthClient(
        settings.discord_client_id,
        settings.discord_client_secret,
        settings.discord_redirect_uri,
        http,
    )
    verifier = DiscordMembershipVerifier(
        oauth,
        settings.discord_guild_id,
        settings.discord_member_role_id,
        settings.admin_role_ids,
        settings.admin_user_ids,
    )
    return AuthenticationService(verifier, get_jwt_codec(), get_user_store())


def get_access(
    authorization: Annotated[str | None, Header()] = None,
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
    x_display_name: Annotated[str | None, Header(alias="X-Display-Name")] = None,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> Access:
    """Resolve the caller's access level.

    Priority:
    1. Valid ``X-Admin-Key`` → admin (full access). An optional
       ``X-Display-Name`` header is used as the admin's display name so
       documents they add are attributed to them (attribution only —
       it never affects authorisation).
    2. Valid ``Authorization: Bearer <JWT>`` → member or admin based on claims.
    3. Valid ``Authorization: Bearer <MEMBER_TOKEN>`` → member.
    4. No credentials → public.
    """
    # 1. Admin API key (kept for backward compatibility and scripts).
    if x_admin_key is not None:
        if _admin_key_valid(x_admin_key):
            name = (x_display_name or "").strip()[:48]
            return Access(is_member=True, is_admin=True, display_name=name or "Admin (API key)")
        raise AuthenticationError("Invalid admin key.")

    # 2. Bearer token: JWT or static member token.
    if credentials is not None:
        token = credentials.credentials

        # Static member token.
        settings = get_settings()
        if settings.member_token and secrets.compare_digest(
            token.encode(), settings.member_token.encode()
        ):
            return Access(is_member=True, display_name="Member (static token)")

        # JWT from Discord OAuth.
        claims = get_jwt_codec().decode(token)
        if claims is not None:
            return Access(
                is_member=True,
                is_admin=claims.is_admin,
                user_id=claims.user_id,
                display_name=claims.display_name,
                avatar=claims.avatar,
            )

    # 3. No valid credentials → public.
    return Access(is_member=False)


def require_admin(access: Annotated[Access, Depends(get_access)]) -> Access:
    """FastAPI dependency: reject unless the caller is an admin."""
    if not access.is_admin:
        raise AuthenticationError("Admin privileges required.")
    return access


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
    return ChatService(
        pipeline=pipeline,
        conversations=conversations,
        query_logs=get_query_log_repository(),
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Ingestion service writing into the same collection the chat reads."""
    return IngestionService(store=get_store(), lexical=get_lexical_index())


@lru_cache(maxsize=1)
def get_feedback_repository() -> FeedbackRepository:
    return FeedbackRepository(get_settings().feedback_db_path)


@lru_cache(maxsize=1)
def get_query_log_repository() -> QueryLogRepository:
    return QueryLogRepository(get_settings().conversation_db_path)
