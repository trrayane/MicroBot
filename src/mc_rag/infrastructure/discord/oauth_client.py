"""Discord OAuth2 client (authorization URL, code exchange, guild member lookup)."""

from __future__ import annotations

import logging
from urllib.parse import urlencode

import httpx

from mc_rag.domain.exceptions import InvalidInput, UpstreamUnavailable

logger = logging.getLogger(__name__)

_API = "https://discord.com/api/v10"
_SCOPES = "identify guilds.members.read"


class DiscordOAuthClient:
    """Thin wrapper around the Discord OAuth2 endpoints."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        http: httpx.AsyncClient,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._http = http

    def authorize_url(self, state: str) -> str:
        """Build the URL that redirects the user to Discord's consent screen."""
        query = urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self._redirect_uri,
                "response_type": "code",
                "scope": _SCOPES,
                "state": state,
            }
        )
        return f"https://discord.com/oauth2/authorize?{query}"

    async def exchange_code(self, code: str) -> str:
        """Exchange an authorization code for a short-lived access token.

        The token is returned to the caller and **never stored**.
        """
        res = await self._send(
            "POST",
            f"{_API}/oauth2/token",
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri,
            },
        )
        if res.status_code in (400, 401):
            raise InvalidInput("Invalid or expired Discord code")
        self._raise_for_upstream(res)
        payload: dict[str, str] = res.json()
        return payload["access_token"]

    async def get_guild_member(
        self, access_token: str, guild_id: str
    ) -> dict[str, object] | None:
        """Return the guild member object for the authenticated user (user token).

        Uses ``GET /users/@me/guilds/{guild_id}/member`` which works with the
        user's OAuth token and does NOT require the bot to be in the server.
        Returns ``None`` when the user is not a member of *guild_id*.
        """
        res = await self._send(
            "GET",
            f"{_API}/users/@me/guilds/{guild_id}/member",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if res.status_code in (403, 404):
            return None
        self._raise_for_upstream(res)
        return dict(res.json())

    async def get_current_user(self, access_token: str) -> dict[str, object]:
        """Return the Discord user object for the given access token."""
        res = await self._send(
            "GET",
            f"{_API}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        self._raise_for_upstream(res)
        return dict(res.json())

    async def get_user_guilds(self, access_token: str) -> list[dict[str, object]]:
        """Return the list of guilds the user is a member of."""
        res = await self._send(
            "GET",
            f"{_API}/users/@me/guilds",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        self._raise_for_upstream(res)
        data = res.json()
        return list(data) if isinstance(data, list) else []

    async def _send(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        try:
            return await self._http.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable("Discord is unreachable") from exc

    @staticmethod
    def _raise_for_upstream(res: httpx.Response) -> None:
        if res.status_code >= 400:
            logger.warning("Discord API error %d: %.200s", res.status_code, res.text)
            raise UpstreamUnavailable("Discord rejected the request")
