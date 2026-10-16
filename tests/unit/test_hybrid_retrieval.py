"""Retriever behaviour when ``HYBRID_ENABLED=true``.

The legacy (vector-only) path must stay untouched when the flag is off, so
every test here states which mode it exercises. Fakes are used for both
indexes: what matters is the orchestration (who is queried, with which
visibility, and how candidates are fused and gated), not the scoring engines
covered by ``test_bm25_index`` / ``test_hybrid`` / ``test_retrieval``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mc_rag.config import Settings
from mc_rag.infrastructure.lexical_store.bm25_index import IndexEntry, LexicalHit
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_PUBLIC,
    SearchResult,
)
from mc_rag.rag.retrieval import Retriever

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStore:
    """Vector store recording every call it receives."""

    def __init__(self, hits: list[SearchResult] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[dict[str, Any]] = []

    def search_by_text(
        self,
        text: str,
        embedder: Any,
        *,
        top_k: int | None = None,
        visibilities: Any = None,
        where: Any = None,
    ) -> list[SearchResult]:
        self.calls.append({"text": text, "top_k": top_k, "visibilities": visibilities})
        limit = len(self.hits) if top_k is None else top_k
        return list(self.hits)[:limit]


class FakeLexical:
    """BM25 index recording every call it receives."""

    def __init__(self, hits: list[LexicalHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[dict[str, Any]] = []

    def search(
        self, query: str, *, visibilities: Any, limit: int = 20
    ) -> list[LexicalHit]:
        self.calls.append({"query": query, "visibilities": visibilities, "limit": limit})
        return list(self.hits)[:limit]


class DummyEmbedder:
    """Never used by the fakes, but the Retriever requires an embedder."""


def vector_hit(chunk_id: str, score: float, *, visibility: str = VISIBILITY_PUBLIC) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"vector text {chunk_id}",
        metadata={"chunk_id": chunk_id, "visibility": visibility, "source": f"{chunk_id}.md"},
        score=score,
    )


def lexical_hit(chunk_id: str, force: float, *, visibility: str = VISIBILITY_PUBLIC) -> LexicalHit:
    return LexicalHit(
        entry=IndexEntry(
            chunk_id=chunk_id,
            text=f"lexical text {chunk_id}",
            metadata={"chunk_id": chunk_id, "visibility": visibility, "source": f"{chunk_id}.md"},
        ),
        bm25_score=-force,
    )


def retriever(
    *,
    store: FakeStore,
    lexical: FakeLexical,
    hybrid: bool = True,
    min_score: float = 0.5,
    bm25_min: float = 1.0,
    require_both: bool = False,
    top_k: int = 10,
    pool: int | None = None,
) -> Retriever:
    """Build a retriever wired to the fakes."""
    return Retriever(
        store=store,  # type: ignore[arg-type]
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        lexical=lexical,  # type: ignore[arg-type]
        top_k=top_k,
        min_score=min_score,
        hybrid_enabled=hybrid,
        rerank_enabled=False,
        hybrid_bm25_min_score=bm25_min,
        hybrid_candidate_pool=pool,
        hybrid_require_both=require_both,
    )


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #


def test_hybrid_is_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYBRID_ENABLED", raising=False)
    assert Settings(_env_file=None).hybrid_enabled is True


def test_disabled_mode_never_queries_the_lexical_index() -> None:
    store = FakeStore([vector_hit("a", 0.9)])
    lexical = FakeLexical([lexical_hit("b", 50.0)])

    result = retriever(store=store, lexical=lexical, hybrid=False).retrieve(
        "question", is_member=False
    )

    assert lexical.calls == []
    assert len(store.calls) == 1
    assert result.hybrid is False
    assert result.best_bm25_score is None
    assert result.rrf_best is None
    assert [hit.chunk_id for hit in result.hits] == ["a"]


def test_disabled_mode_still_drops_hits_below_the_threshold() -> None:
    store = FakeStore([vector_hit("a", 0.2)])
    lexical = FakeLexical([lexical_hit("b", 50.0)])

    result = retriever(store=store, lexical=lexical, hybrid=False, min_score=0.5).retrieve(
        "question", is_member=False
    )

    assert result.has_answer is False
    assert result.best_score == 0.2  # raw best kept for observability


# --------------------------------------------------------------------------- #
# Fusion and gating
# --------------------------------------------------------------------------- #


def test_hybrid_fuses_both_indexes() -> None:
    store = FakeStore([vector_hit("v1", 0.9), vector_hit("shared", 0.8)])
    lexical = FakeLexical([lexical_hit("shared", 12.0), lexical_hit("l1", 9.0)])

    result = retriever(store=store, lexical=lexical).retrieve("question", is_member=False)

    assert result.hybrid is True
    assert [hit.chunk_id for hit in result.hits] == ["shared", "v1", "l1"]
    assert result.best_score == 0.9
    assert result.best_bm25_score == 12.0
    assert result.rrf_best is not None and result.rrf_best > 0


def test_hybrid_keeps_a_strong_lexical_hit_below_the_vector_threshold() -> None:
    store = FakeStore([vector_hit("below", 0.1)])
    lexical = FakeLexical([lexical_hit("term", 25.0)])

    result = retriever(store=store, lexical=lexical, min_score=0.7).retrieve(
        "question", is_member=False
    )

    # The cosine is far too low, but the BM25 hit alone justifies keeping it;
    # the vector candidate fails both gates and is dropped.
    assert {hit.chunk_id for hit in result.hits} == {"term"}


def test_hybrid_drops_candidates_failing_both_indexes() -> None:
    store = FakeStore([vector_hit("weak", 0.1)])
    lexical = FakeLexical([lexical_hit("weak", 0.01)])

    result = retriever(store=store, lexical=lexical, min_score=0.7, bm25_min=1.0).retrieve(
        "question", is_member=False
    )

    assert result.hits == []
    assert result.has_answer is False
    assert result.best_score == 0.1
    assert result.best_bm25_score == pytest.approx(0.01)


def test_hybrid_require_both_keeps_only_double_matches() -> None:
    store = FakeStore([vector_hit("both", 0.9), vector_hit("vector-only", 0.9)])
    lexical = FakeLexical([lexical_hit("both", 12.0), lexical_hit("lex-only", 12.0)])

    result = retriever(
        store=store, lexical=lexical, require_both=True, bm25_min=1.0, min_score=0.5
    ).retrieve("question", is_member=False)

    assert [hit.chunk_id for hit in result.hits] == ["both"]


def test_hybrid_without_any_candidate_has_no_answer() -> None:
    store = FakeStore([])
    lexical = FakeLexical([])

    result = retriever(store=store, lexical=lexical).retrieve("question", is_member=False)

    assert result.hits == []
    assert result.has_answer is False
    assert result.best_score is None
    assert result.best_bm25_score is None
    assert result.rrf_best is None


def test_hybrid_bounds_hits_by_top_k() -> None:
    store = FakeStore([vector_hit(f"v{i}", 0.9 - i * 0.01) for i in range(8)])
    lexical = FakeLexical([lexical_hit(f"l{i}", 20.0 - i) for i in range(8)])

    result = retriever(store=store, lexical=lexical, top_k=3, min_score=0.0).retrieve(
        "question", is_member=False
    )

    assert len(result.hits) == 3


def test_hybrid_pool_never_drops_below_top_k() -> None:
    store = FakeStore([vector_hit(f"v{i}", 0.9) for i in range(5)])
    lexical = FakeLexical([lexical_hit(f"l{i}", 5.0) for i in range(5)])

    retriever(store=store, lexical=lexical, top_k=10, pool=2).retrieve(
        "question", is_member=False
    )

    assert store.calls[0]["top_k"] == 10
    assert lexical.calls[0]["limit"] == 10


# --------------------------------------------------------------------------- #
# Visibility and query construction
# --------------------------------------------------------------------------- #


def test_public_caller_restricts_both_indexes_to_public() -> None:
    store = FakeStore([])
    lexical = FakeLexical([])

    retriever(store=store, lexical=lexical).retrieve("question", is_member=False)

    assert store.calls[0]["visibilities"] == (VISIBILITY_PUBLIC,)
    assert lexical.calls[0]["visibilities"] == (VISIBILITY_PUBLIC,)


def test_member_caller_queries_both_indexes_with_all_visibilities() -> None:
    store = FakeStore([])
    lexical = FakeLexical([])

    retriever(store=store, lexical=lexical).retrieve("question", is_member=True)

    assert store.calls[0]["visibilities"] == ALL_VISIBILITIES
    assert lexical.calls[0]["visibilities"] == ALL_VISIBILITIES


def test_hybrid_uses_the_follow_up_search_query() -> None:
    store = FakeStore([])
    lexical = FakeLexical([])
    history = [{"role": "user", "content": "Quand est le hackathon ?"}]

    result = retriever(store=store, lexical=lexical).retrieve(
        "et où ?", is_member=False, history=history
    )

    assert result.search_query == "Quand est le hackathon ? et où ?"
    assert store.calls[0]["text"] == result.search_query
    assert lexical.calls[0]["query"] == result.search_query
