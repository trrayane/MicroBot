"""Tests for chat streaming, auto-titles, follow-ups and feedback reasons."""

from __future__ import annotations

from pathlib import Path

import pytest

from mc_rag.application.chat_service import ChatService
from mc_rag.application.conversation_service import ConversationService
from mc_rag.infrastructure.llm.openrouter_client import LLMResponse, LLMStreamChunk
from mc_rag.infrastructure.persistence.conversation_repository import (
    ConversationRepository,
)
from mc_rag.infrastructure.persistence.feedback_repository import FeedbackRepository
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository
from mc_rag.rag.pipeline import RagPipeline
from mc_rag.rag.retrieval import RetrievalResult


class FakeRetriever:
    def __init__(self, hits: list) -> None:
        self.hits = hits

    def retrieve(self, message, *, is_member, history=None) -> RetrievalResult:
        return RetrievalResult(search_query=message, hits=self.hits, best_score=0.9)


class FakeLLM:
    """Scripted LLM: one reply for the answer, one for the title/followups."""

    def __init__(self, answer: str = "Voici la réponse [1].") -> None:
        self.answer = answer
        self.answer_calls = 0
        self.titles: list[str] = []
        self.followups: list[str] = []

    def complete(self, messages):
        prompt = messages[-1]["content"]
        if "short chat title" in prompt:
            self.titles.append(prompt)
            return LLMResponse(text="Titre simple", prompt_tokens=10, completion_tokens=5)
        if "follow-up" in prompt:
            self.followups.append(prompt)
            return LLMResponse(
                text="Comment faire ?\nQuelles sont les règles ?\nOù s'inscrire ?",
                prompt_tokens=20,
                completion_tokens=15,
            )
        self.answer_calls += 1
        return LLMResponse(text=self.answer, prompt_tokens=100, completion_tokens=25)


class FakeStreamLLM(FakeLLM):
    """Same as FakeLLM but streams the answer as two deltas."""

    def stream(self, messages):
        yield LLMStreamChunk(text="Voici la réponse ")
        yield LLMStreamChunk(text="[1].")
        yield LLMStreamChunk(prompt_tokens=100, completion_tokens=25, final=True)


def make_hit():
    from mc_rag.infrastructure.vector_store.chroma_store import SearchResult

    return SearchResult(
        chunk_id="c1",
        text="Texte du document",
        metadata={"doc_id": "d1", "source": "guide.pdf", "title": "Guide"},
        distance=0.1,
        score=0.9,
    )


def make_service(tmp_path: Path, llm=None, *, followups_enabled: bool = True) -> ChatService:
    conversations = ConversationService(ConversationRepository(str(tmp_path / "conv.db")))
    query_logs = QueryLogRepository(str(tmp_path / "queries.db"))
    pipeline = RagPipeline(
        retriever=FakeRetriever([make_hit()]),
        llm=llm or FakeLLM(),
        followups_enabled=followups_enabled,
        followups_count=3,
    )
    return ChatService(pipeline=pipeline, conversations=conversations, query_logs=query_logs)


# --------------------------------------------------------------------------- #
# Auto title
# --------------------------------------------------------------------------- #


def test_first_turn_generates_and_persists_title(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    outcome = service.chat(
        "Comment rejoindre MC ?", conversation_id=None, is_member=False
    )
    assert outcome.title == "Titre simple"
    # The title is persisted: a second call on the same conversation reuses it.
    again = service.chat("Et après ?", conversation_id=outcome.conversation_id, is_member=False)
    assert again.title == "Titre simple"
    # Only one title call was made (the second turn reuses the stored one).
    assert len(service._pipeline._llm.titles) == 1  # noqa: SLF001


def test_title_failure_does_not_break_the_chat(tmp_path: Path) -> None:
    class BrokenTitleLLM(FakeLLM):
        def complete(self, messages):
            prompt = messages[-1]["content"]
            if "short chat title" in prompt:
                raise RuntimeError("boom")
            return super().complete(messages)

    service = make_service(tmp_path, BrokenTitleLLM())
    outcome = service.chat("Une question ?", conversation_id=None, is_member=False)
    assert outcome.title is None
    assert outcome.result.grounded is True


def test_conversation_title_survives_reopen(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    outcome = service.chat("Ma question ?", conversation_id=None, is_member=False)
    _, history, title = service._conversations.open(  # noqa: SLF001
        outcome.conversation_id, is_member=False
    )
    assert title == "Titre simple"
    assert history[-1]["role"] == "assistant"


# --------------------------------------------------------------------------- #
# Follow-ups
# --------------------------------------------------------------------------- #


def test_followups_are_generated_for_grounded_answers(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    outcome = service.chat("Comment rejoindre MC ?", conversation_id=None, is_member=False)
    assert outcome.result.followups == [
        "Comment faire",
        "Quelles sont les règles",
        "Où s'inscrire",
    ]


def test_no_followups_when_disabled(tmp_path: Path) -> None:
    service = make_service(tmp_path, followups_enabled=False)
    outcome = service.chat("Comment rejoindre MC ?", conversation_id=None, is_member=False)
    assert outcome.result.followups == []


def test_no_followups_on_refusal(tmp_path: Path) -> None:
    from mc_rag.rag.grounding import refusal_message

    conversations = ConversationService(ConversationRepository(str(tmp_path / "c.db")))
    query_logs = QueryLogRepository(str(tmp_path / "q.db"))
    pipeline = RagPipeline(
        retriever=FakeRetriever([]),  # no hits -> refusal, no LLM call
        llm=FakeLLM(),
        followups_enabled=True,
    )
    service = ChatService(
        pipeline=pipeline, conversations=conversations, query_logs=query_logs
    )
    outcome = service.chat("Quelle est la capitale du Japon ?", conversation_id=None, is_member=False)
    assert outcome.result.grounded is False
    assert outcome.result.answer == refusal_message("fr")
    assert outcome.result.followups == []


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_chat_stream_yields_deltas_then_done(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeStreamLLM())
    events = list(
        service.chat_stream("Comment rejoindre MC ?", conversation_id=None, is_member=False)
    )
    deltas = [e.delta for e in events if e.type == "delta"]
    assert deltas == ["Voici la réponse ", "[1]."]
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    payload = done[0].payload
    assert payload["answer"] == "Voici la réponse [1]."
    assert payload["grounded"] is True
    assert payload["title"] == "Titre simple"
    assert payload["followups"] == [
        "Comment faire",
        "Quelles sont les règles",
        "Où s'inscrire",
    ]
    assert payload["sources"][0]["doc_id"] == "d1"
    assert payload["query_log_id"] > 0
    assert payload["conversation_id"]


def test_chat_stream_persists_the_turn(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeStreamLLM())
    events = list(
        service.chat_stream("Ma question ?", conversation_id=None, is_member=False)
    )
    done = next(e for e in events if e.type == "done")
    cid = done.payload["conversation_id"]
    # The turn was stored: reopening gives history with the question.
    _, history, title = service._conversations.open(cid, is_member=False)  # noqa: SLF001
    assert history[0] == {"role": "user", "content": "Ma question ?"}
    assert title == "Titre simple"


def test_stream_without_stream_method_falls_back_to_complete(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeLLM())  # no .stream method
    events = list(service.chat_stream("Une question ?", conversation_id=None, is_member=False))
    deltas = [e.delta for e in events if e.type == "delta"]
    assert deltas == ["Voici la réponse [1]."]
    assert any(e.type == "done" for e in events)


# --------------------------------------------------------------------------- #
# Feedback with reason
# --------------------------------------------------------------------------- #


def test_feedback_stores_reason(tmp_path: Path) -> None:
    repo = FeedbackRepository(str(tmp_path / "fb.db"))
    record = repo.add(
        conversation_id="c_1", vote="down", reason="  Je cherchais les dates du hackathon  "
    )
    assert record.reason == "Je cherchais les dates du hackathon"
    listed = repo.list()
    assert listed[0].reason == "Je cherchais les dates du hackathon"


def test_feedback_reason_is_optional_and_capped(tmp_path: Path) -> None:
    repo = FeedbackRepository(str(tmp_path / "fb.db"))
    none = repo.add(conversation_id="c_1", vote="up")
    assert none.reason is None
    long = repo.add(conversation_id="c_1", vote="down", reason="x" * 1000)
    assert long.reason is not None and len(long.reason) == 500


def test_feedback_migration_adds_reason_column(tmp_path: Path) -> None:
    import sqlite3
    from contextlib import closing

    db = tmp_path / "fb.db"
    # Create an old-schema DB without the reason column.
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE feedback (id TEXT PRIMARY KEY, conversation_id TEXT, "
            "vote TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('f_old', 'c_1', 'up', 1.0)"
        )
    repo = FeedbackRepository(str(db))
    listed = repo.list()
    assert listed[0].feedback_id == "f_old"
    assert listed[0].reason is None
    # New writes work with the reason column.
    repo.add(conversation_id="c_2", vote="down", reason="ok")
    assert repo.list()[0].reason == "ok"


# --------------------------------------------------------------------------- #
# Conversation title column migration
# --------------------------------------------------------------------------- #


def test_conversation_title_column_migration(tmp_path: Path) -> None:
    import sqlite3
    from contextlib import closing

    db = tmp_path / "conv.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, "
            "is_member INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO conversations VALUES ('c_old', 0, 1.0)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
    repo = ConversationRepository(str(db))
    record = repo.get("c_old")
    assert record is not None
    assert record.title is None
    repo.set_title("c_old", "Mon titre")
    assert repo.get("c_old").title == "Mon titre"
