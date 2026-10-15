"""Cross-encoder reranker for retrieval precision.

First-stage retrieval (vector + BM25) optimises for *recall*: it fetches a
wide candidate pool, some of which are only loosely related. A cross-encoder
reads the query and each candidate *jointly* (full attention between the two
texts), producing a much more accurate relevance score. It is the single most
effective way to improve precision after a good first stage.

Model choice: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80 MB, loads in ~2 s)
is the lightest useful cross-encoder. It is English-first but handles FR/AR
queries well enough for re-scoring because the first-stage retriever already
did the multilingual heavy lifting. Swap to ``BAAI/bge-reranker-v2-m3`` (~2 GB)
for stronger multilingual precision when latency budget allows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Candidates above this cosine-style logit are considered relevant.
DEFAULT_RERANK_MIN_SCORE = 0.0
# How many candidates the cross-encoder scores (before cutting to top_k).
DEFAULT_RERANK_POOL = 20


@runtime_checkable
class _CrossEncoderLike(Protocol):
    """Minimal surface of the sentence-transformers CrossEncoder API."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> object: ...


class CrossEncoderReranker:
    """Reranks retrieval candidates with a cross-encoder model.

    The reranker takes ``(query, chunk_text)`` pairs, scores them, and returns
    the chunk indices sorted best-first. Candidates whose score falls below
    ``min_score`` are dropped entirely, so the LLM never sees weakly relevant
    context.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        min_score: float | None = None,
        pool_size: int | None = None,
        model: _CrossEncoderLike | None = None,
    ) -> None:
        from mc_rag.config import get_settings

        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.min_score = min_score if min_score is not None else settings.rerank_min_score
        self.pool_size = pool_size if pool_size is not None else settings.rerank_pool_size
        self._model = model

    @property
    def model(self) -> _CrossEncoderLike:
        """The underlying cross-encoder, loaded on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Score every text against *query* and return ``(index, score)`` pairs.

        Indices refer to the original ``texts`` ordering. Results are sorted by
        decreasing score and truncated to ``top_k``. Candidates below
        ``min_score`` are excluded.
        """
        if not texts or top_k <= 0:
            return []

        pairs = [(query, text) for text in texts]
        try:
            raw = self.model.predict(pairs)
        except Exception:  # noqa: BLE001 - reranking is best-effort
            logger.exception("Cross-encoder reranking failed; falling back to original order.")
            return [(i, 0.0) for i in range(min(top_k, len(texts)))]

        scores = _to_float_list(raw)
        ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])
        kept = [(idx, score) for idx, score in ranked if score >= self.min_score]
        return kept[:top_k]

    def _load_model(self) -> _CrossEncoderLike:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is required at runtime
            raise RuntimeError(
                "sentence-transformers is not installed; install the project "
                "dependencies to use the reranker."
            ) from exc
        logger.info("Loading cross-encoder reranker %s", self.model_name)
        return CrossEncoder(self.model_name)


def _to_float_list(raw: object) -> list[float]:
    """Normalise the CrossEncoder predict output to a flat list of floats."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()  # type: ignore[union-attr]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        result: list[float] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                result.extend(_to_float_list(item))
            else:
                result.append(float(item))
        return result
    return [float(raw)]  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Return a process-wide cached :class:`CrossEncoderReranker`."""
    return CrossEncoderReranker()
