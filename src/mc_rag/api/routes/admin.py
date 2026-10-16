"""Admin reporting endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from mc_rag.api.dependencies import get_query_log_repository, require_admin
from mc_rag.api.schemas import RepetitiveQuestionOut, RepetitiveQuestionsResponse
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