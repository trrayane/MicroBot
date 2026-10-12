"""API error handlers: every error is JSON ``{"error": {"code", "message"}}``.

Messages for infrastructure failures are fixed strings so internals (stack
traces, provider details, keys) never reach the client.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mc_rag.domain.exceptions import (
    AuthenticationError,
    ConversationNotFoundError,
    EmbeddingError,
    LLMRateLimitError,
    LLMUnavailableError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

# exception type -> (status, code, client message or None to use str(exc))
_DOMAIN_ERRORS: list[tuple[type[Exception], int, str, str | None]] = [
    (AuthenticationError, 401, "unauthorized", None),
    (ConversationNotFoundError, 404, "conversation_not_found", None),
    (
        LLMRateLimitError,
        429,
        "rate_limited",
        "The assistant reached its usage limit. Please retry later.",
    ),
    (
        LLMUnavailableError,
        503,
        "llm_unavailable",
        "The language model is temporarily unavailable. Please retry later.",
    ),
    (VectorStoreError, 503, "service_unavailable", "The knowledge base is unavailable."),
    (EmbeddingError, 503, "service_unavailable", "The knowledge base is unavailable."),
]

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def register_error_handlers(app: FastAPI) -> None:
    def make_handler(
        status: int, code: str, message: str | None
    ) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
        async def handler(_: Request, exc: Exception) -> JSONResponse:
            if status >= 500:
                logger.warning("%s: %s", code, exc)
            return error_response(status, code, message or str(exc))

        return handler

    for exc_type, status, code, message in _DOMAIN_ERRORS:
        app.add_exception_handler(exc_type, make_handler(status, code, message))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "Invalid request.")
        message = f"Invalid request: {where}: {detail}" if where else f"Invalid request: {detail}"
        return error_response(422, "validation_error", message)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return error_response(500, "internal_error", "An unexpected error occurred.")