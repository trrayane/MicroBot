"""Domain-level models shared across layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiscordMember:
    """A verified Discord guild member."""

    user_id: str
    display_name: str
    roles: tuple[str, ...]
    is_admin: bool = False
    avatar: str | None = None
