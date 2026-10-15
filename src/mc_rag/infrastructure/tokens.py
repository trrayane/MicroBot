"""JWT codec for stateless session tokens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt


@dataclass(frozen=True, slots=True)
class SessionClaims:
    """Decoded JWT payload."""

    user_id: str
    display_name: str
    is_admin: bool = False
    avatar: str | None = None


class JwtCodec:
    """Issue and decode HS256 session tokens."""

    def __init__(self, secret: str, ttl_minutes: int = 720) -> None:
        self._secret = secret
        self._ttl = timedelta(minutes=ttl_minutes)

    def issue(
        self,
        user_id: str,
        display_name: str,
        is_admin: bool = False,
        avatar: str | None = None,
    ) -> str:
        """Create a signed token for the given user."""
        now = datetime.now(UTC)
        payload: dict[str, object] = {
            "sub": user_id,
            "name": display_name,
            "admin": is_admin,
            "iat": now,
            "exp": now + self._ttl,
        }
        if avatar:
            payload["avatar"] = avatar
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def decode(self, token: str) -> SessionClaims | None:
        """Decode *token*; returns ``None`` on any failure."""
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError:
            return None
        return SessionClaims(
            user_id=payload["sub"],
            display_name=payload.get("name", ""),
            is_admin=payload.get("admin") is True,
            avatar=payload.get("avatar"),
        )
