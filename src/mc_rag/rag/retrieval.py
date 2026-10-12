"""Vector retrieval: turns a user message into the relevant, allowed chunks.

This stage makes **zero LLM calls**. It does three things:

1. Build the text to search with (follow-up questions borrow context from the
   previous user message, no query rewriting by an LLM).
2. Search Chroma with the visibilities the caller is allowed to see. The filter
   is applied inside the vector store, never in the prompt.
3. Drop hits whose score is below ``min_score``. When nothing is left,
   ``has_answer`` is ``False`` and the pipeline can refuse without calling the LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from mc_rag.config import get_settings
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder, get_embedder
from mc_rag.infrastructure.vector_store.chroma_store import (
    ChromaStore,
    SearchResult,
    allowed_visibilities,
)

logger = logging.getLogger(__name__)

# A message this short ("and when is it?") is probably a follow-up.
FOLLOW_UP_MAX_WORDS = 6


@dataclass(slots=True)
class RetrievalResult:
    """Outcome of a retrieval step."""

    search_query: str
    hits: list[SearchResult] = field(default_factory=list)
    best_score: float | None = None  # best raw score, even if below threshold

    @property
    def has_answer(self) -> bool:
        """Whether at least one chunk passed the score threshold."""
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
    """Searches the vector store on behalf of a caller with a given access level."""

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        embedder: LocalEmbedder | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or ChromaStore()
        self._embedder = embedder or get_embedder()
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.min_score = min_score if min_score is not None else settings.retrieval_min_score

    def retrieve(
        self,
        message: str,
        *,
        is_member: bool,
        history: Sequence[dict[str, str]] | None = None,
    ) -> RetrievalResult:
        """Return the allowed chunks relevant to *message*."""
        search_query = build_search_query(message, history)
        raw_hits = self._store.search_by_text(
            search_query,
            self._embedder,
            top_k=self.top_k,
            visibilities=allowed_visibilities(is_member=is_member),
        )
        kept = [hit for hit in raw_hits if hit.score >= self.min_score]
        best = max((hit.score for hit in raw_hits), default=None)
        logger.info(
            "retrieval query_len=%d member=%s raw=%d kept=%d best_score=%s",
            len(search_query),
            is_member,
            len(raw_hits),
            len(kept),
            best,
        )
        return RetrievalResult(search_query=search_query, hits=kept, best_score=best)