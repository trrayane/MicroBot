"""Retriever + rewriting/multi-query: the budget and the fusion rules.

The contract under test:

* **zero** LLM call when the first pass is already good (no history);
* **zero** LLM call when the first pass found nothing at all;
* **one** call when the first pass is weak or the question is a follow-up;
* every extra query (rewritten + variants) is searched and fused;
* the pipeline spends at most **two** calls in total (rewriting + answer).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from mc_rag.infrastructure.lexical_store.bm25_index import IndexEntry, LexicalHit
from mc_rag.infrastructure.vector_store.chroma_store import SearchResult
from mc_rag.rag.pipeline import ChatResult, RagPipeline
from mc_rag.rag.query_rewriting import QueryPlan
from mc_rag.rag.retrieval import Retriever

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStore:
    """Vector store serving hits per query text."""

    def __init__(self, per_query: dict[str, list[SearchResult]] | None = None) -> None:
        self.per_query = per_query or {}
        self.calls: list[str] = []

    def search_by_text(
        self,
        text: str,
        embedder: Any,
        *,
        top_k: int | None = None,
        visibilities: Any = None,
        where: Any = None,
    ) -> list[SearchResult]:
        self.calls.append(text)
        hits = self.per_query.get(text, [])
        return list(hits) if top_k is None else list(hits)[:top_k]


class FakeLexical:
    """BM25 index serving hits per query text."""

    def __init__(self, per_query: dict[str, list[LexicalHit]] | None = None) -> None:
        self.per_query = per_query or {}
        self.calls: list[str] = []

    def search(
        self, query: str, *, visibilities: Any, limit: int = 20
    ) -> list[LexicalHit]:
        self.calls.append(query)
        return list(self.per_query.get(query, []))[:limit]


@dataclass(slots=True)
class FakeRewriter:
    """Rewriter double returning a fixed plan, recording who asked for it."""

    plan_result: QueryPlan = field(default_factory=QueryPlan)
    calls: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True

    def plan(self, message: str, *, history: Sequence[dict[str, str]] | None = None) -> QueryPlan:
        self.calls.append({"message": message, "history": history})
        return self.plan_result


class DummyEmbedder:
    """Never used by the fakes, but the Retriever requires an embedder."""


@dataclass(slots=True)
class FakeLLMResponse:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5


class FakeLLM:
    """Chat client double for the pipeline budget test."""

    def __init__(self, text: str = "[[SOURCE 1]] answer") -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages: Sequence[dict[str, str]]) -> FakeLLMResponse:
        self.calls += 1
        return FakeLLMResponse(text=self.text)


def vector_hit(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"chunk_id": chunk_id, "visibility": "public", "source": f"{chunk_id}.md"},
        score=score,
    )


def lexical_hit(chunk_id: str, force: float) -> LexicalHit:
    return LexicalHit(
        entry=IndexEntry(
            chunk_id=chunk_id,
            text=f"lexical {chunk_id}",
            metadata={"chunk_id": chunk_id, "visibility": "public", "source": f"{chunk_id}.md"},
        ),
        bm25_score=-force,
    )


def retriever(
    *,
    store: FakeStore,
    lexical: FakeLexical | None = None,
    rewriter: FakeRewriter | None = None,
    hybrid: bool = False,
    min_score: float = 0.5,
    top_k: int = 10,
) -> Retriever:
    """Build a retriever wired to the fakes."""
    return Retriever(
        store=store,  # type: ignore[arg-type]
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        lexical=lexical,  # type: ignore[arg-type]
        rewriter=rewriter,  # type: ignore[arg-type]
        top_k=top_k,
        min_score=min_score,
        hybrid_enabled=hybrid,
        rerank_enabled=False,
    )


# --------------------------------------------------------------------------- #
# When the call is (not) spent
# --------------------------------------------------------------------------- #


def test_no_rewriter_means_no_extra_queries_and_no_calls() -> None:
    store = FakeStore({"question": [vector_hit("a", 0.9)]})

    result = retriever(store=store).retrieve("question", is_member=False)

    assert store.calls == ["question"]
    assert result.llm_calls == 0
    assert result.queries_run == 1
    assert result.rewritten is None


def test_strong_first_pass_without_history_spends_no_call() -> None:
    store = FakeStore({"question": [vector_hit("a", 0.9)]})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))

    result = retriever(store=store, rewriter=rw, min_score=0.5).retrieve(
        "a long enough question about the club", is_member=False
    )

    assert rw.calls == []
    assert store.calls == ["a long enough question about the club"]
    assert result.llm_calls == 0
    assert result.queries_run == 1
    assert result.rewritten is None


def test_weak_first_pass_spends_exactly_one_call() -> None:
    store = FakeStore({"question": [vector_hit("a", 0.2)]})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))

    result = retriever(store=store, rewriter=rw, min_score=0.5).retrieve(
        "question", is_member=False
    )

    assert len(rw.calls) == 1
    assert result.llm_calls == 1
    assert result.queries_run == 2  # original + rewritten
    assert result.rewritten == "rewritten"
    assert store.calls == ["question", "rewritten"]


def test_history_always_spends_the_call_even_on_a_strong_first_pass() -> None:
    store = FakeStore({"question et le reste": [vector_hit("a", 0.9)]})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))
    history = [{"role": "user", "content": "question"}]

    result = retriever(store=store, rewriter=rw).retrieve(
        "et le reste", is_member=False, history=history
    )

    assert len(rw.calls) == 1
    assert rw.calls[0]["history"] == history
    assert result.llm_calls == 1


def test_first_pass_without_any_candidate_refuses_without_a_call() -> None:
    store = FakeStore({})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))

    result = retriever(store=store, rewriter=rw).retrieve("question", is_member=False)

    assert rw.calls == []  # 0 LLM call: the pipeline will refuse
    assert store.calls == ["question"]  # one search only
    assert result.has_answer is False
    assert result.llm_calls == 0
    assert result.queries_run == 1


def test_failed_rewriting_call_is_counted_even_without_a_plan() -> None:
    store = FakeStore({"question": [vector_hit("a", 0.2)]})
    rw = FakeRewriter(plan_result=QueryPlan(llm_calls=1))  # call spent, nothing usable

    result = retriever(store=store, rewriter=rw).retrieve("question", is_member=False)

    assert result.llm_calls == 1
    assert result.queries_run == 1  # only the original was searched
    assert result.rewritten is None


# --------------------------------------------------------------------------- #
# Extra queries are searched and fused
# --------------------------------------------------------------------------- #


def test_variants_are_searched_and_fused_with_the_original() -> None:
    store = FakeStore(
        {
            "question": [vector_hit("weak", 0.2)],
            "rewritten": [vector_hit("good", 0.9)],
            "variant one": [vector_hit("variant", 0.8)],
        }
    )
    rw = FakeRewriter(
        plan_result=QueryPlan(rewritten="rewritten", variants=("variant one",), llm_calls=1)
    )

    result = retriever(store=store, rewriter=rw, hybrid=False, min_score=0.5).retrieve(
        "question", is_member=False
    )

    assert store.calls == ["question", "rewritten", "variant one"]
    assert result.queries_run == 3
    assert result.llm_calls == 1
    # Fused by RRF; "weak" (0.2) is below the threshold and dropped in
    # vector-only mode.
    assert [hit.chunk_id for hit in result.hits] == ["good", "variant"]


def test_rewritten_query_is_used_in_hybrid_mode_too() -> None:
    store = FakeStore(
        {"question": [vector_hit("weak", 0.2)], "rewritten": [vector_hit("strong", 0.9)]}
    )
    lexical = FakeLexical({"question": [], "rewritten": [lexical_hit("lex", 12.0)]})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))

    result = retriever(store=store, lexical=lexical, rewriter=rw, hybrid=True).retrieve(
        "question", is_member=False
    )

    assert result.hybrid is True
    assert lexical.calls == ["question", "rewritten"]
    assert result.llm_calls == 1
    assert {"strong", "lex"} <= {hit.chunk_id for hit in result.hits}


def test_extra_query_hits_below_the_threshold_are_still_dropped() -> None:
    store = FakeStore(
        {
            "question": [vector_hit("a", 0.2)],
            "rewritten": [vector_hit("b", 0.3)],
        }
    )
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))

    result = retriever(store=store, rewriter=rw, min_score=0.5).retrieve(
        "question", is_member=False
    )

    assert result.hits == []
    assert result.best_score == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Pipeline budget: rewriting + answer <= 2 calls
# --------------------------------------------------------------------------- #


def test_pipeline_reports_both_llm_calls() -> None:
    store = FakeStore({"question": [vector_hit("a", 0.2)], "rewritten": [vector_hit("a", 0.9)]})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))
    llm = FakeLLM(text="[0] grounded answer")

    pipeline = RagPipeline(retriever=retriever(store=store, rewriter=rw), llm=llm)
    result: ChatResult = pipeline.answer("question", is_member=False)

    assert rw.calls and llm.calls == 1
    assert result.llm_calls == 2  # one rewriting + one answer, never more
    assert result.grounded is True


def test_pipeline_refusal_spends_no_call() -> None:
    store = FakeStore({})
    rw = FakeRewriter(plan_result=QueryPlan(rewritten="rewritten", llm_calls=1))
    llm = FakeLLM()

    pipeline = RagPipeline(retriever=retriever(store=store, rewriter=rw), llm=llm)
    result = pipeline.answer("question", is_member=False)

    assert rw.calls == []
    assert llm.calls == 0
    assert result.llm_calls == 0
    assert result.grounded is False
