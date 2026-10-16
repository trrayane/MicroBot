"""Aggregates and mounts all API route modules under ``/api/v1``."""

from __future__ import annotations

from fastapi import APIRouter

from mc_rag.api.routes import admin, auth, chat, documents, feedback, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(admin.router)
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(feedback.router)