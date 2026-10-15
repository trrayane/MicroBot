"""Retrieval: turns a user message into the relevant, allowed chunks.

It does three things:

1. Build the text to search with (follow-up questions borrow context from the
   previous user message).
2. Optionally spend **one** LLM call to rewrite the question and/or expand it
   into variants, when ``QUERY_REWRITE_ENABLED`` / ``MULTI_QUERY_ENABLED`` are
   on and the first pass looked weak (or there is a conversation history).
3. Search the indexes with the visibilities the caller is allowed to see. The
   filter is applied **inside** each index (Chroma ``where`` clause, SQL
   ``WHERE``), never in the prompt.

Two modes, selected by ``HYBRID_ENABLED`` (default ``false`` = legacy):

* **vector only** - Chroma cosine scores, keep ``score >= threshold``;
* **hybrid** - Chroma *and* SQLite FTS5 (BM25) candidates fused with RRF, then
  the accept rule of :mod:`mc_rag.rag.hybrid` (strong vector hit **or** strong
  lexical hit, both when ``HYBRID_REQUIRE_BOTH``).

When nothing passes the relevance gate, ``has_answer`` is ``False`` and the
pipeline can refuse without calling the LLM. Total cost stays at most
**two** LLM calls per question (one rewriting, one answer).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from mc_rag.config import get_settings
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder, get_embedder
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index, LexicalHit
from mc_rag.infrastructure.llm.client import get_llm_client
from mc_rag.infrastructure.vector_store.chroma_store import (
    ChromaStore,
    SearchResult,
    allowed_visibilities,
)
from mc_rag.rag.grounding import detect_language
from mc_rag.rag.hybrid import (
    Candidate,
    accept,
    candidate_from_lexical,
    candidate_from_vector,
    rrf_fuse,
    to_search_result,
)
from mc_rag.rag.query_rewriting import QueryPlan, QueryRewriter

logger = logging.getLogger(__name__)

# A message this short ("and when is it?") is probably a follow-up.
FOLLOW_UP_MAX_WORDS = 6


@dataclass(slots=True)
class RetrievalResult:
    """Outcome of a retrieval step."""

    search_query: str
    hits: list[SearchResult] = field(default_factory=list)
    best_score: float | None = None  # best raw cosine, even if below threshold
    # Hybrid observability (both stay ``None`` when hybrid is disabled).
    hybrid: bool = False
    best_bm25_score: float | None = None  # best -bm25, even if below threshold
    rrf_best: float | None = None  # best fused score actually kept (or proposed)
    # Rewriting / multi-query observability.
    llm_calls: int = 0  # rewriting calls spent (0 or 1), the answer call is separate
    queries_run: int = 1  # queries actually searched, original included
    rewritten: str | None = None  # the rewritten query, when one was used

    @property
    def has_answer(self) -> bool:
        """Whether at least one chunk passed the relevance gate."""
        return bool(self.hits)


def build_search_query(message: str, history: Sequence[dict[str, str]] | None = None) -> str:
    """Return the text to embed for *message*.

    History items look like ``{"role": "user" | "assistant", "content": "..."}``.
    A short message is prefixed with the previous user message so that
    "and when is it?" still points at the right topic.
    """
    message = message.strip()
    if not history or len(message.split()) > FOLLOW_UP_MAX_WORDS:
        return message
    for turn in reversed(history):
        if turn.get("role") == "user" and turn.get("content", "").strip():
            return f"{turn['content'].strip()} {message}"
    return message


class Retriever:
    """Searches the indexes on behalf of a caller with a given access level."""

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        embedder: LocalEmbedder | None = None,
        lexical: BM25Index | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        min_score_multilingual: float | None = None,
        hybrid_enabled: bool | None = None,
        hybrid_bm25_min_score: float | None = None,
        hybrid_candidate_pool: int | None = None,
        hybrid_require_both: bool | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or ChromaStore()
        self._embedder = embedder or get_embedder()
        self._lexical = lexical
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        # An explicit ``min_score`` overrides every language (useful for tests).
        self._min_score_override = min_score
        self.min_score = min_score if min_score is not None else settings.retrieval_min_score
        self.min_score_multilingual = (
            min_score_multilingual
            if min_score_multilingual is not None
            else settings.retrieval_min_score_multilingual
        )
        # Hybrid retrieval: off by default, never required by the legacy path.
        self.hybrid_enabled = (
            settings.hybrid_enabled if hybrid_enabled is None else hybrid_enabled
        )
        self.hybrid_rrf_k = settings.hybrid_rrf_k
        self.hybrid_bm25_min_score = (
            settings.hybrid_bm25_min_score
            if hybrid_bm25_min_score is None
            else hybrid_bm25_min_score
        )
        self.hybrid_require_both = (
            settings.hybrid_require_both
            if hybrid_require_both is None
            else hybrid_require_both
        )
        pool = settings.hybrid_candidate_pool if hybrid_candidate_pool is None else (
            hybrid_candidate_pool
        )
        self.hybrid_candidate_pool = max(pool, self.top_k)
        # Query rewriting / multi-query: off by default, one shared LLM call.
        if rewriter is None and (
            settings.query_rewrite_enabled or settings.multi_query_enabled
        ):
            rewriter = QueryRewriter(get_llm_client())
        self.rewriter = rewriter

    @property
    def lexical(self) -> BM25Index:
        """The BM25 index, opened lazily and only used in hybrid mode."""
        if self._lexical is None:
            self._lexical = BM25Index()
        return self._lexical

    def _threshold_for(self, message: str) -> float:
        """Return the score threshold to apply for *message*'s language.

        English uses ``min_score``; French and Arabic use the lower
        ``min_score_multilingual`` because cross-lingual cosine scores against
        English sources are compressed. An explicit override wins everywhere.
        """
        if self._min_score_override is not None:
            return self._min_score_override
        if detect_language(message) == "en":
            return self.min_score
        return self.min_score_multilingual

    def retrieve(
        self,
        message: str,
        *,
        is_member: bool,
        history: Sequence[dict[str, str]] | None = None,
    ) -> RetrievalResult:
        """Return the allowed chunks relevant to *message*."""
        base_query = build_search_query(message, history)
        threshold = self._threshold_for(message)
        visibilities = allowed_visibilities(is_member=is_member)
        pool = self.hybrid_candidate_pool if self.hybrid_enabled else self.top_k

        vector_hits = self._store.search_by_text(
            base_query,
            self._embedder,
            top_k=pool,
            visibilities=visibilities,
        )
        lexical_hits = (
            self.lexical.search(base_query, visibilities=visibilities, limit=pool)
            if self.hybrid_enabled
            else []
        )

        plan = self._maybe_plan(
            message, history, vector_hits, lexical_hits, threshold=threshold
        )
        extra = plan.queries
        vector_lists = [vector_hits]
        lexical_lists: list[list[LexicalHit]] = [lexical_hits] if self.hybrid_enabled else []
        for query in extra:
            vector_lists.append(
                self._store.search_by_text(
                    query,
                    self._embedder,
                    top_k=pool,
                    visibilities=visibilities,
                )
            )
            if self.hybrid_enabled:
                lexical_lists.append(
                    self.lexical.search(query, visibilities=visibilities, limit=pool)
                )

        if self.hybrid_enabled:
            return self._assemble_hybrid(
                base_query,
                vector_lists=vector_lists,
                lexical_lists=lexical_lists,
                threshold=threshold,
                is_member=is_member,
                plan=plan,
            )
        if not extra:
            return self._assemble_legacy(
                base_query,
                vector_hits,
                threshold=threshold,
                is_member=is_member,
                plan=plan,
            )
        return self._assemble_vector_multi(
            base_query,
            vector_lists=vector_lists,
            threshold=threshold,
            is_member=is_member,
            plan=plan,
        )

    def _maybe_plan(
        self,
        message: str,
        history: Sequence[dict[str, str]] | None,
        vector_hits: Sequence[SearchResult],
        lexical_hits: Sequence[LexicalHit],
        *,
        threshold: float,
    ) -> QueryPlan:
        """Spend the (at most one) rewriting call, only when it can help.

        Rules: zero call when the first pass found nothing at all (the pipeline
        will refuse), zero when it already cleared the threshold with no
        history, one call otherwise (weak first pass or follow-up question).

        ``QUERY_REWRITE_MIN_SCORE`` overrides the relevance threshold for this
        decision (``0.0`` keeps the language threshold).
        """
        if self.rewriter is None or not self.rewriter.active:
            return QueryPlan()
        if not vector_hits and not lexical_hits:
            return QueryPlan()
        settings = get_settings()
        gate = threshold if settings.query_rewrite_min_score <= 0.0 else (
            settings.query_rewrite_min_score
        )
        best = max((hit.score for hit in vector_hits), default=None)
        if best is not None and best >= gate and not history:
            return QueryPlan()
        return self.rewriter.plan(message, history=history)

    def _assemble_legacy(
        self,
        base_query: str,
        vector_hits: list[SearchResult],
        *,
        threshold: float,
        is_member: bool,
        plan: QueryPlan,
    ) -> RetrievalResult:
        """Vector-only, single query: the historical behaviour, byte for byte."""
        kept = [hit for hit in vector_hits if hit.score >= threshold]
        best = max((hit.score for hit in vector_hits), default=None)
        logger.info(
            "retrieval query_len=%d member=%s raw=%d kept=%d threshold=%.2f best_score=%s",
            len(base_query),
            is_member,
            len(vector_hits),
            len(kept),
            threshold,
            best,
        )
        return RetrievalResult(
            search_query=base_query,
            hits=kept,
            best_score=best,
            llm_calls=plan.llm_calls,
            rewritten=plan.rewritten or None,
        )

    def _assemble_vector_multi(
        self,
        base_query: str,
        *,
        vector_lists: Sequence[Sequence[SearchResult]],
        threshold: float,
        is_member: bool,
        plan: QueryPlan,
    ) -> RetrievalResult:
        """Vector-only, several queries fused with RRF (rewriting / multi-query)."""
        rankings = [[candidate_from_vector(hit) for hit in hits] for hits in vector_lists]
        fused = rrf_fuse(rankings, k=self.hybrid_rrf_k)
        kept = [
            item
            for item in fused
            if item.candidate.vector_score is not None
            and item.candidate.vector_score >= threshold
        ][: self.top_k]
        hits = [to_search_result(item.candidate) for item in kept]
        best = max(
            (hit.score for hits in vector_lists for hit in hits), default=None
        )
        logger.info(
            "retrieval query_len=%d member=%s queries=%d raw=%d kept=%d "
            "threshold=%.2f best_score=%s rewritten=%s",
            len(base_query),
            is_member,
            len(vector_lists),
            sum(len(hits_) for hits_ in vector_lists),
            len(kept),
            threshold,
            best,
            bool(plan.rewritten),
        )
        return RetrievalResult(
            search_query=base_query,
            hits=hits,
            best_score=best,
            llm_calls=plan.llm_calls,
            queries_run=len(vector_lists),
            rewritten=plan.rewritten or None,
        )

    def _assemble_hybrid(
        self,
        base_query: str,
        *,
        vector_lists: Sequence[Sequence[SearchResult]],
        lexical_lists: Sequence[Sequence[LexicalHit]],
        threshold: float,
        is_member: bool,
        plan: QueryPlan,
    ) -> RetrievalResult:
        """Fuse every vector and BM25 ranking (RRF) then apply the accept rule."""
        rankings: list[list[Candidate]] = [
            [candidate_from_vector(hit) for hit in hits] for hits in vector_lists
        ]
        rankings.extend(
            [candidate_from_lexical(hit) for hit in hits] for hits in lexical_lists
        )
        fused = rrf_fuse(rankings, k=self.hybrid_rrf_k)

        kept = [
            item
            for item in fused
            if accept(
                item.candidate,
                threshold=threshold,
                bm25_min=self.hybrid_bm25_min_score,
                require_both=self.hybrid_require_both,
            )
        ][: self.top_k]

        hits = [to_search_result(item.candidate) for item in kept]
        best_score = max((hit.score for hits_ in vector_lists for hit in hits_), default=None)
        best_bm25 = max(
            (
                candidate.bm25_force
                for ranking in rankings
                for candidate in ranking
                if candidate.bm25_force is not None
            ),
            default=None,
        )
        rrf_best = fused[0].rrf_score if fused else None
        logger.info(
            "hybrid retrieval query_len=%d member=%s queries=%d vector=%d lexical=%d "
            "fused=%d kept=%d threshold=%.2f bm25_min=%.2f best_score=%s "
            "best_bm25=%s rrf=%s llm_calls=%d",
            len(base_query),
            is_member,
            len(vector_lists),
            sum(len(hits_) for hits_ in vector_lists),
            sum(len(hits_) for hits_ in lexical_lists),
            len(fused),
            len(hits),
            threshold,
            self.hybrid_bm25_min_score,
            best_score,
            best_bm25,
            rrf_best,
            plan.llm_calls,
        )
        return RetrievalResult(
            search_query=base_query,
            hits=hits,
            best_score=best_score,
            hybrid=True,
            best_bm25_score=best_bm25,
            rrf_best=rrf_best,
            llm_calls=plan.llm_calls,
            queries_run=len(vector_lists),
            rewritten=plan.rewritten or None,
        )