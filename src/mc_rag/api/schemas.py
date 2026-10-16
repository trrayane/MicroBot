"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

MAX_MESSAGE_CHARS = 2000
MAX_TITLE_CHARS = 200


class ChatRequest(BaseModel):
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)
    ]
    conversation_id: str | None = None


class SourceOut(BaseModel):
    doc_id: str
    title: str
    snippet: str


class UsageOut(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    query_log_id: int
    grounded: bool
    sources: list[SourceOut]
    usage: UsageOut
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str


class StatsResponse(BaseModel):
    collection: str
    documents: int
    chunks: int


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    source: str
    visibility: str
    chunks: int
    embedding_model: str
    uploaded_by: str | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut]


class AddDocumentResponse(BaseModel):
    doc_id: str
    title: str
    status: str  # processing | indexed | updated | unchanged | failed
    chunks: int
    visibility: str


class DocumentStatusResponse(BaseModel):
    doc_id: str
    title: str
    status: str
    chunks: int = 0
    error: str | None = None
    elapsed_ms: int = 0
    # Where the job is right now (parsing | chunking | embedding | indexing)
    # and how far along the chunk work is, so a client can show real progress.
    stage: str = "queued"
    chunks_done: int = 0
    chunks_total: int = 0


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


class LoginRequest(BaseModel):
    key: Annotated[str, StringConstraints(min_length=1)] = ""


class LoginResponse(BaseModel):
    ok: bool


class MeResponse(BaseModel):
    role: str
    display_name: str | None = None
    avatar: str | None = None
    user_id: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str | None = None
    query_log_id: int | None = None
    vote: Literal["up", "down"]


class FeedbackOut(BaseModel):
    feedback_id: str
    conversation_id: str | None
    vote: str
    created_at: float


class FeedbackListResponse(BaseModel):
    feedback: list[FeedbackOut]


class RepetitiveQuestionOut(BaseModel):
    normalized_query: str
    total_count: int
    unuseful_count: int


class RepetitiveQuestionsResponse(BaseModel):
    questions: list[RepetitiveQuestionOut]