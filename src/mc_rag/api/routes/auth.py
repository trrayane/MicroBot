"""Authentication endpoints: admin key login + Discord OAuth."""

from __future__ import annotations

import hmac
import secrets
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from mc_rag.api.dependencies import (
    Access,
    get_access,
    get_authentication_service,
    get_user_store,
)
from mc_rag.api.schemas import LoginRequest, LoginResponse, MeResponse
from mc_rag.application.authentication_service import AuthenticationService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import AuthenticationError
from mc_rag.infrastructure.persistence.user_store import UserStore

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory OAuth state store (avoids cookie issues when the user accesses
# the site via both 127.0.0.1 and localhost — cookies are per-host).
_pending_states: dict[str, float] = {}
_STATE_TTL = 600.0  # seconds


def _store_state(state: str) -> None:
    now = time.monotonic()
    # purge expired entries
    for k in [k for k, exp in _pending_states.items() if now > exp]:
        del _pending_states[k]
    _pending_states[state] = now + _STATE_TTL


def _consume_state(state: str) -> bool:
    exp = _pending_states.pop(state, None)
    return exp is not None and time.monotonic() <= exp


# --------------------------------------------------------------------------- #
# Admin API key (kept for backward compatibility)
# --------------------------------------------------------------------------- #


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """Verify the admin key; the client then sends it as ``X-Admin-Key``."""
    expected = get_settings().admin_api_key
    if not expected or not hmac.compare_digest(body.key.encode(), expected.encode()):
        raise AuthenticationError("Invalid admin key.")
    return LoginResponse(ok=True)


# --------------------------------------------------------------------------- #
# Discord OAuth2
# --------------------------------------------------------------------------- #


@router.get("/discord/login", include_in_schema=False)
async def discord_login(
    auth: Annotated[AuthenticationService, Depends(get_authentication_service)],
) -> RedirectResponse:
    """Redirect the user to Discord's consent screen."""
    state = secrets.token_urlsafe(24)
    _store_state(state)
    return RedirectResponse(auth.login_url(state))


@router.get("/discord/callback", include_in_schema=False)
async def discord_callback(
    auth: Annotated[AuthenticationService, Depends(get_authentication_service)],
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Handle the Discord redirect: validate state, exchange code, issue JWT."""

    def back(fragment: str) -> RedirectResponse:
        return RedirectResponse(f"/#{fragment}", status_code=302)

    if error:  # user cancelled at Discord
        return back("error=access_denied")

    if not (code and state and _consume_state(state)):
        return back("error=invalid_state")

    token = await auth.login(code)
    return back("error=not_member" if token is None else f"token={token}")


@router.get("/me", response_model=MeResponse)
async def me(access: Annotated[Access, Depends(get_access)]) -> MeResponse:
    """Return the current user's identity and role."""
    if access.user_id is None and not access.is_admin:
        return MeResponse(role="anonymous", display_name=None, avatar=None, user_id=None)

    role = "admin" if access.is_admin else ("member" if access.is_member else "anonymous")
    return MeResponse(
        role=role,
        display_name=access.display_name,
        avatar=access.avatar,
        user_id=access.user_id,
    )


@router.get("/users", include_in_schema=False)
async def list_users(
    access: Annotated[Access, Depends(get_access)],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> dict[str, object]:
    """Return all known Discord users (admin only)."""
    if not access.is_admin:
        raise AuthenticationError("Admin privileges required.")
    return {"users": store.list_users()}
