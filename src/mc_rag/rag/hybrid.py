"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) over vector and BM25 hits.

Fusion
------
Each input ranking is a list of :class:`Candidate` ordered best-first (the
vector ranking and the lexical ranking, later also one ranking per rewritten
query). RRF scores every candidate with::

    score = sum(1 / (k + rank_i))     for each ranking containing the candidate

with ``k = HYBRID_RRF_K`` (60 by default). RRF only uses **ranks**, so the
incomparable scales of cosine similarity and BM25 never meet, and a candidate
appearing in both lists naturally outranks a single-list one.

Accept rule
-----------
Fusion only ranks; the *decision* to keep a candidate stays explicit:

* the candidate is a strong vector hit (cosine >= the language threshold), **or**
* it is a strong lexical hit (``-bm25 >= HYBRID_BM25_MIN_SCORE``);

when ``HYBRID_REQUIRE_BOTH`` is ``true`` both must hold instead. With hybrid
disabled the vector rule alone applies, which is exactly the legacy behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mc_rag.infrastructure.lexical_store.bm25_index import LexicalHit
from mc_rag.infrastructure.vector_store.chroma_store import SearchResult


@dataclass(slots=True, frozen=True)
class Candidate:
    """A chunk proposed by one or more rankings."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float | None = None  # cosine similarity, when available
    bm25_force: float | None = None  # -bm25, higher is a stronger match

    @property
    def is_vector_hit(self) -> bool:
        """Whether the candidate came out of the vector ranking."""
        return self.vector_score is not None

    @property
    def is_lexical_hit(self) -> bool:
        """Whether the candidate came out of the BM25 ranking."""
        return self.bm25_force is not None


@dataclass(slots=True, frozen=True)
class FusedCandidate:
    """A candidate with its fused rank score."""

    candidate: Candidate
    rrf_score: float


def candidate_from_vector(hit: SearchResult) -> Candidate:
    """Wrap a Chroma hit as a :class:`Candidate`."""
    return Candidate(
        chunk_id=hit.chunk_id,
        text=hit.text,
        metadata=dict(hit.metadata),
        vector_score=hit.score,
    )


def candidate_from_lexical(hit: LexicalHit) -> Candidate:
    """Wrap a BM25 hit as a :class:`Candidate`."""
    return Candidate(
        chunk_id=hit.entry.chunk_id,
        text=hit.entry.text,
        metadata=dict(hit.entry.metadata),
        bm25_force=hit.force,
    )


def rrf_fuse(
    rankings: Sequence[Sequence[Candidate]],
    *,
    k: int = 60,
) -> list[FusedCandidate]:
    """Fuse *rankings* (each ordered best-first) with Reciprocal Rank Fusion.

    Duplicates inside a single ranking keep their best (lowest) rank. Ties are
    broken by ``chunk_id`` so the result is deterministic.

    Args:
        rankings: One ordered candidate list per source.
        k: RRF dampening constant (``HYBRID_RRF_K``); must be > 0.

    Returns:
        Fused candidates ordered by decreasing RRF score.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    scores: dict[str, float] = {}
    merged: dict[str, Candidate] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, candidate in enumerate(ranking, start=1):
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (k + rank)
            merged[candidate.chunk_id] = _merge(merged.get(candidate.chunk_id), candidate)

    fused = [
        FusedCandidate(candidate=merged[chunk_id], rrf_score=score)
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item.rrf_score, item.candidate.chunk_id))
    return fused


def accept(
    candidate: Candidate,
    *,
    threshold: float,
    bm25_min: float,
    require_both: bool = False,
) -> bool:
    """Whether *candidate* passes the relevance gate.

    Args:
        candidate: The fused candidate to judge.
        threshold: Minimum cosine similarity for a vector hit.
        bm25_min: Minimum ``-bm25`` for a lexical hit.
        require_both: ``True`` demands the candidate to be strong in **both**
            indexes instead of in either one.
    """
    vector_ok = candidate.vector_score is not None and candidate.vector_score >= threshold
    bm25_ok = candidate.bm25_force is not None and candidate.bm25_force >= bm25_min
    if require_both:
        return vector_ok and bm25_ok
    return vector_ok or bm25_ok


def to_search_result(candidate: Candidate) -> SearchResult:
    """Convert an accepted candidate back into the shared :class:`SearchResult`.

    ``score`` keeps the cosine similarity when there is one (a purely lexical
    hit has no meaningful cosine and reports ``0.0``); the RRF score used for
    ordering is not mixed into it, so the legacy meaning of ``score`` holds.
    """
    return SearchResult(
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        metadata=dict(candidate.metadata),
        distance=0.0,
        score=candidate.vector_score if candidate.vector_score is not None else 0.0,
    )


def _merge(existing: Candidate | None, incoming: Candidate) -> Candidate:
    """Merge two views of the same chunk, keeping the best scores."""
    if existing is None:
        return incoming
    vector_score = existing.vector_score
    if incoming.vector_score is not None:
        vector_score = (
            incoming.vector_score
            if vector_score is None
            else max(vector_score, incoming.vector_score)
        )
    bm25_force = existing.bm25_force
    if incoming.bm25_force is not None:
        bm25_force = (
            incoming.bm25_force if bm25_force is None else max(bm25_force, incoming.bm25_force)
        )
    return Candidate(
        chunk_id=existing.chunk_id,
        text=existing.text or incoming.text,
        metadata=existing.metadata or incoming.metadata,
        vector_score=vector_score,
        bm25_force=bm25_force,
    )
