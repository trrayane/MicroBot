"""Chat endpoint backed by the RAG pipeline."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from mc_rag.api.dependencies import Access, get_access, get_chat_service
from mc_rag.api.schemas import ChatRequest, ChatResponse, SourceOut, UsageOut
from mc_rag.application.chat_service import ChatService

router = APIRouter(tags=["chat"])


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
    result = outcome.result
    return ChatResponse(
        answer=result.answer,
        conversation_id=outcome.conversation_id,
        query_log_id=outcome.query_log_id,
        grounded=result.grounded,
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