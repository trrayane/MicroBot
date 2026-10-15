"""Authentication service: Discord OAuth login + session token issuance."""

from __future__ import annotations

import logging
from typing import Protocol

from mc_rag.domain.models import DiscordMember
from mc_rag.infrastructure.persistence.user_store import UserStore

logger = logging.getLogger(__name__)


class MembershipVerifier(Protocol):
    """Minimal surface of the Discord membership verifier."""

    def authorize_url(self, state: str) -> str: ...

    async def verify(self, code: str) -> DiscordMember | None: ...


class TokenIssuer(Protocol):
    """Minimal surface of the JWT codec."""

    def issue(
        self,
        user_id: str,
        display_name: str,
        is_admin: bool = False,
        avatar: str | None = None,
    ) -> str: ...


class AuthenticationService:
    """Orchestrates Discord login: verify membership, persist user, issue JWT."""

    def __init__(
        self,
        verifier: MembershipVerifier,
        issue_token: TokenIssuer,
        user_store: UserStore,
    ) -> None:
        self._verifier = verifier
        self._issuer = issue_token
        self._users = user_store

    def login_url(self, state: str) -> str:
        """Return the Discord consent URL for *state*."""
        return self._verifier.authorize_url(state)

    async def login(self, code: str) -> str | None:
        """Exchange *code* for a session token, or ``None`` if not a member."""
        member = await self._verifier.verify(code)
        if member is None:
            return None

        # Persist / update the user so the admin table can show uploaders.
        self._users.upsert_user(
            user_id=member.user_id,
            username=member.display_name.lower().replace(" ", "_"),
            display_name=member.display_name,
            avatar=member.avatar,
            is_admin=member.is_admin,
        )

        return self._issuer.issue(
            member.user_id,
            member.display_name,
            member.is_admin,
            member.avatar,
        )
