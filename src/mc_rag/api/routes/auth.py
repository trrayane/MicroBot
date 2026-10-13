"""Admin authentication endpoint (key verification)."""

from __future__ import annotations

import hmac

from fastapi import APIRouter

from mc_rag.api.schemas import LoginRequest, LoginResponse
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import AuthenticationError

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """Verify the admin key; the client then sends it as ``X-Admin-Key``."""
    expected = get_settings().admin_api_key
    if not expected or not hmac.compare_digest(body.key.encode(), expected.encode()):
        raise AuthenticationError("Invalid admin key.")
    return LoginResponse(ok=True)
