"""Health, readiness and knowledge-base statistics endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from mc_rag.api.dependencies import get_store, require_admin
from mc_rag.api.schemas import HealthResponse, StatsResponse
from mc_rag.config import get_settings
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; never touches the model or the vector store."""
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, env=settings.app_env)


@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_admin)])
def stats(store: Annotated[ChromaStore, Depends(get_store)]) -> StatsResponse:
    """High-level document/chunk counts for the admin dashboard."""
    data = store.stats()
    return StatsResponse(**data)
