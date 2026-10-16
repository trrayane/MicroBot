"""Feedback endpoints: submit a vote (public) and list them (admin)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from mc_rag.api.dependencies import get_feedback_repository, get_query_log_repository, require_admin
from mc_rag.api.schemas import FeedbackListResponse, FeedbackOut, FeedbackRequest
from mc_rag.infrastructure.persistence.feedback_repository import FeedbackRepository
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut)
def submit_feedback(
    body: FeedbackRequest,
    repo: Annotated[FeedbackRepository, Depends(get_feedback_repository)],
    query_logs: Annotated[QueryLogRepository, Depends(get_query_log_repository)],
) -> FeedbackOut:
    """Record a thumbs up/down for an answer."""
    record = repo.add(conversation_id=body.conversation_id, vote=body.vote)
    query_logs.set_feedback(
        body.vote,
        query_log_id=body.query_log_id,
        conversation_id=body.conversation_id,
    )
    return FeedbackOut(
        feedback_id=record.feedback_id,
        conversation_id=record.conversation_id,
        vote=record.vote,
        created_at=record.created_at,
    )


@router.get("", response_model=FeedbackListResponse, dependencies=[Depends(require_admin)])
def list_feedback(
    repo: Annotated[FeedbackRepository, Depends(get_feedback_repository)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> FeedbackListResponse:
    """Return the most recent feedback entries (admin only)."""
    records = repo.list(limit=limit)
    return FeedbackListResponse(
        feedback=[
            FeedbackOut(
                feedback_id=record.feedback_id,
                conversation_id=record.conversation_id,
                vote=record.vote,
                created_at=record.created_at,
            )
            for record in records
        ]
    )
