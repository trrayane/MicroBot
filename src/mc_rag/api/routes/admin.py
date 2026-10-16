"""Admin reporting endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from mc_rag.api.dependencies import get_llm, get_query_log_repository, require_admin
from mc_rag.api.schemas import (
    LlmKeyStatus,
    LlmStatusResponse,
    RepetitiveQuestionOut,
    RepetitiveQuestionsResponse,
)
from mc_rag.infrastructure.llm.gemini_client import GeminiClient
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/repetitive-questions", response_model=RepetitiveQuestionsResponse)
def repetitive_questions(
    repository: Annotated[QueryLogRepository, Depends(get_query_log_repository)],
    min_frequency: Annotated[int, Query(ge=1)] = 10,
) -> RepetitiveQuestionsResponse:
    """List normalized questions that meet the requested frequency threshold."""
    questions = repository.list_repetitive(min_frequency=min_frequency)
    return RepetitiveQuestionsResponse(
        questions=[
            RepetitiveQuestionOut(
                normalized_query=question.normalized_query,
                total_count=question.total_count,
                unuseful_count=question.unuseful_count,
            )
            for question in questions
        ]
    )


@router.get("/llm-status", response_model=LlmStatusResponse)
def llm_status(llm: Annotated[GeminiClient, Depends(get_llm)]) -> LlmStatusResponse:
    """Live status of the LLM provider and every key of the roulette pool."""
    if isinstance(llm, GeminiClient):
        return LlmStatusResponse(
            provider="gemini",
            model=llm.model,
            keys=[LlmKeyStatus(**row) for row in llm.pool.snapshot()],
        )
    return LlmStatusResponse(provider="openrouter", model=llm.model, keys=[])