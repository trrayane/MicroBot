"""Aggregates and mounts all API route modules under ``/api/v1``."""

from __future__ import annotations

from fastapi import APIRouter

from mc_rag.api.routes import chat

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat.router)