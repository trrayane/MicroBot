"""Chat endpoints backed by the RAG pipeline (plain JSON + SSE streaming)."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from mc_rag.api.dependencies import Access, get_access, get_chat_service
from mc_rag.api.schemas import ChatRequest, ChatResponse, SourceOut, UsageOut
from mc_rag.application.chat_service import ChatService
from mc_rag.domain.exceptions import (
    ConversationNotFoundError,
    LLMRateLimitError,
    LLMUnavailableError,
)

router = APIRouter(tags=["chat"])


def _chat_response(
    outcome,
) -> ChatResponse:
    result = outcome.result
    return ChatResponse(
        answer=result.answer,
        conversation_id=outcome.conversation_id,
        query_log_id=outcome.query_log_id,
        grounded=result.grounded,
        title=outcome.title,
        followups=result.followups,
        sources=[
            SourceOut(doc_id=s.doc_id, title=s.title, snippet=s.snippet) for s in result.sources
        ],
        usage=UsageOut(
            llm_calls=result.llm_calls,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        ),
        latency_ms=result.latency_ms,
    )


# Plain ``def``: the pipeline is blocking, FastAPI runs it in a worker thread.
@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    access: Annotated[Access, Depends(get_access)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    outcome = service.chat(
        body.message, conversation_id=body.conversation_id, is_member=access.is_member
    )
    return _chat_response(outcome)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    access: Annotated[Access, Depends(get_access)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Events: ``{"type":"delta","delta":"..."}`` while the model writes, then a
    final ``{"type":"done", ...}`` carrying the same payload as ``/chat``.
    Errors after the stream opened are reported as
    ``{"type":"error","code":...,"message":...}``.
    """

    def event_stream():
        try:
            for event in service.chat_stream(
                body.message, conversation_id=body.conversation_id, is_member=access.is_member
            ):
                if event.type == "delta":
                    yield _sse({"type": "delta", "delta": event.delta})
                elif event.type == "done" and event.payload is not None:
                    yield _sse({"type": "done", **event.payload})
        except ConversationNotFoundError as exc:
            yield _sse({"type": "error", "code": "conversation_not_found", "message": str(exc)})
        except LLMRateLimitError:
            yield _sse(
                {
                    "type": "error",
                    "code": "rate_limited",
                    "message": "The assistant reached its usage limit. Please retry later.",
                }
            )
        except LLMUnavailableError:
            yield _sse(
                {
                    "type": "error",
                    "code": "llm_unavailable",
                    "message": "The language model is temporarily unavailable. Please retry later.",
                }
            )
        except Exception:  # noqa: BLE001 - never leak internals into the stream
            yield _sse(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
