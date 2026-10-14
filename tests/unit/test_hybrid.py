"""Unit tests for the hybrid rank fusion (RRF) and the accept rule."""

from __future__ import annotations

import pytest

from mc_rag.infrastructure.vector_store.chroma_store import SearchResult
from mc_rag.rag.hybrid import (
    Candidate,
    accept,
    candidate_from_vector,
    rrf_fuse,
    to_search_result,
)


def vector_candidate(chunk_id: str, score: float) -> Candidate:
    """A candidate produced by the vector ranking."""
    return Candidate(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"chunk_id": chunk_id, "visibility": "public"},
        vector_score=score,
    )


def lexical_candidate(chunk_id: str, force: float) -> Candidate:
    """A candidate produced by the BM25 ranking."""
    return Candidate(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"chunk_id": chunk_id, "visibility": "public"},
        bm25_force=force,
    )


def test_rrf_prefers_candidates_found_in_both_rankings() -> None:
    fused = rrf_fuse(
        [
            [vector_candidate("v-only", 0.9), vector_candidate("both", 0.8)],
            [lexical_candidate("both", 12.0), lexical_candidate("l-only", 9.0)],
        ],
        k=60,
    )

    assert [item.candidate.chunk_id for item in fused] == ["both", "v-only", "l-only"]
    both = fused[0]
    # 1/(60+2) from the vector list + 1/(60+1) from the lexical list.
    assert both.rrf_score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_uses_ranks_so_incompatible_scales_never_meet() -> None:
    # Cosines around 0.6 and BM25 forces around 15 are not comparable: only
    # the order matters.
    fused = rrf_fuse(
        [
            [vector_candidate("a", 0.61), vector_candidate("b", 0.60)],
            [lexical_candidate("b", 15.0), lexical_candidate("a", 1.5)],
        ],
        k=1,
    )

    assert {item.candidate.chunk_id for item in fused} == {"a", "b"}
    assert fused[0].rrf_score == fused[1].rrf_score


def test_rrf_merges_scores_of_a_chunk_listed_twice() -> None:
    fused = rrf_fuse(
        [
            [vector_candidate("c1", 0.9), lexical_candidate("c1", 8.0)],
            [lexical_candidate("c1", 8.0)],
        ],
        k=60,
    )

    (only,) = fused
    assert only.candidate.vector_score == 0.9
    assert only.candidate.bm25_force == 8.0


def test_rrf_ignores_duplicates_within_a_single_ranking() -> None:
    fused = rrf_fuse(
        [[vector_candidate("c1", 0.9), vector_candidate("c1", 0.5)]],
        k=60,
    )

    (only,) = fused
    assert only.rrf_score == pytest.approx(1 / 61)  # counted once, at rank 1
    assert only.candidate.vector_score == 0.9  # best score kept


def test_rrf_ties_are_broken_deterministically() -> None:
    rankings = [[vector_candidate("a", 0.5)], [vector_candidate("b", 0.5)]]

    assert [i.candidate.chunk_id for i in rrf_fuse(rankings, k=60)] == ["a", "b"]
    assert [i.candidate.chunk_id for i in rrf_fuse(rankings, k=60)] == ["a", "b"]


def test_rrf_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        rrf_fuse([[vector_candidate("c1", 1.0)]], k=0)


def test_rrf_of_no_rankings_is_empty() -> None:
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []


# --------------------------------------------------------------------------- #
# Accept rule
# --------------------------------------------------------------------------- #


def test_accept_keeps_a_strong_vector_hit_even_without_lexical_match() -> None:
    candidate = vector_candidate("c1", 0.75)

    assert accept(candidate, threshold=0.6, bm25_min=1.0) is True


def test_accept_keeps_a_strong_lexical_hit_even_with_a_weak_cosine() -> None:
    candidate = Candidate(chunk_id="c1", text="t", vector_score=0.1, bm25_force=15.0)

    assert accept(candidate, threshold=0.6, bm25_min=1.0) is True


def test_accept_keeps_a_purely_lexical_hit() -> None:
    candidate = lexical_candidate("c1", 15.0)

    assert accept(candidate, threshold=0.6, bm25_min=1.0) is True


def test_accept_drops_a_candidate_failing_both_indexes() -> None:
    candidate = Candidate(chunk_id="c1", text="t", vector_score=0.2, bm25_force=0.1)

    assert accept(candidate, threshold=0.6, bm25_min=1.0) is False


def test_accept_requires_both_when_configured() -> None:
    strong_vector = vector_candidate("c1", 0.9)
    strong_lexical = lexical_candidate("c2", 15.0)
    both = Candidate(chunk_id="c3", text="t", vector_score=0.9, bm25_force=15.0)

    assert accept(strong_vector, threshold=0.6, bm25_min=1.0, require_both=True) is False
    assert accept(strong_lexical, threshold=0.6, bm25_min=1.0, require_both=True) is False
    assert accept(both, threshold=0.6, bm25_min=1.0, require_both=True) is True


def test_accept_never_keeps_a_candidate_without_scores() -> None:
    assert accept(Candidate(chunk_id="c1", text="t"), threshold=0.0, bm25_min=0.0) is False


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #


def test_candidate_from_vector_carries_the_cosine() -> None:
    hit = SearchResult(chunk_id="c1", text="hello", metadata={"visibility": "public"}, score=0.42)

    candidate = candidate_from_vector(hit)

    assert candidate.chunk_id == "c1"
    assert candidate.vector_score == 0.42
    assert candidate.bm25_force is None
    assert candidate.is_vector_hit and not candidate.is_lexical_hit


def test_to_search_result_keeps_the_cosine_and_reports_lexical_only_as_zero() -> None:
    with_cosine = to_search_result(vector_candidate("c1", 0.8))
    lexical_only = to_search_result(lexical_candidate("c2", 12.0))

    assert with_cosine.score == 0.8
    assert with_cosine.text == "text c1"
    assert lexical_only.score == 0.0
    assert lexical_only.text == "text c2"
    assert lexical_only.metadata["visibility"] == "public"


def test_to_search_result_does_not_leak_the_rrf_score() -> None:
    fused = rrf_fuse([[vector_candidate("c1", 0.33)]], k=60)

    result = to_search_result(fused[0].candidate)

    assert result.score == 0.33
    assert result.distance == 0.0
