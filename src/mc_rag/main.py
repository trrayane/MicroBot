"""Application entrypoint: builds and runs the FastAPI app."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mc_rag.api.errors import register_error_handlers
from mc_rag.api.router import api_router
from mc_rag.config import get_settings
from mc_rag.infrastructure.embeddings.local_embedder import get_embedder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model once at startup so the first request is fast.

    The warm-up runs in a thread so uvicorn can still serve requests
    (Discord OAuth callback, health, …) while the model loads.
    """

    def _warm_up() -> None:
        try:
            get_embedder().embed_query("warm up")
            logger.info("Embedding model ready.")
        except Exception:  # noqa: BLE001 - the app can still start; requests will report it
            logger.exception("Embedding model warm-up failed.")

    asyncio.get_running_loop().run_in_executor(None, _warm_up)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stash the shared httpx client for the Discord OAuth client.
    app.state.http = httpx.AsyncClient(timeout=8.0)

    app.include_router(api_router)
    register_error_handlers(app)

    # Serve the single-page frontend. Mounted last so /api/v1/* wins.
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()