"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, StringConstraints

MAX_MESSAGE_CHARS = 2000


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
    grounded: bool
    sources: list[SourceOut]
    usage: UsageOut
    latency_ms: int