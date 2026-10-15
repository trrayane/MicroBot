"""Verify Discord guild membership and role-based admin access."""

from __future__ import annotations

import logging

from mc_rag.domain.models import DiscordMember
from mc_rag.infrastructure.discord.oauth_client import DiscordOAuthClient

logger = logging.getLogger(__name__)


class DiscordMembershipVerifier:
    """Turns an OAuth code into a :class:`DiscordMember` (or ``None``)."""

    def __init__(
        self,
        oauth: DiscordOAuthClient,
        guild_id: str,
        required_role_id: str = "",
        admin_role_ids: frozenset[str] = frozenset(),
        admin_user_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._oauth = oauth
        self._guild_id = guild_id
        self._required_role = required_role_id or None
        self._admin_roles = admin_role_ids
        self._admin_users = admin_user_ids

    def authorize_url(self, state: str) -> str:
        """Build the Discord consent URL for *state*."""
        return self._oauth.authorize_url(state)

    async def verify(self, code: str) -> DiscordMember | None:
        """Exchange *code*, check guild membership and roles.

        Returns:
            The verified member, or ``None`` when the user is not in the
            guild (or lacks the required role).
        """
        access_token = await self._oauth.exchange_code(code)  # never stored

        # Get the user identity from the OAuth token.
        user_data = await self._oauth.get_current_user(access_token)
        user_id = str(user_data.get("id", ""))
        if not user_id:
            return None

        # Check guild membership + get roles via the user token.
        # Uses GET /users/@me/guilds/{guild_id}/member which does NOT require
        # the bot to be in the server.
        member_data = await self._oauth.get_guild_member(access_token, self._guild_id)
        if member_data is None:
            logger.info("discord login user=%s: not in guild", user_id)
            return None

        raw_roles = member_data.get("roles", [])
        roles = tuple(str(r) for r in raw_roles) if isinstance(raw_roles, list) else ()
        is_admin = bool(self._admin_roles & set(roles)) or user_id in self._admin_users
        logger.info("discord login user=%s admin=%s", user_id, is_admin)

        # Admins bypass the required-role check.
        if not is_admin and self._required_role and self._required_role not in roles:
            return None

        nick = member_data.get("nick")
        global_name = user_data.get("global_name")
        username = user_data.get("username", "")
        name = str(nick or global_name or username)
        avatar = user_data.get("avatar")
        return DiscordMember(
            user_id=user_id,
            display_name=name,
            roles=roles,
            is_admin=is_admin,
            avatar=str(avatar) if avatar else None,
        )
