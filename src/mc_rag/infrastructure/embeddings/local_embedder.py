"""Local embedding adapter built on ``sentence-transformers``.

The heavy model import and loading happen lazily on first use, so importing this
module (or the package it belongs to) never requires ``torch`` to be present.
The model itself can be injected at construction time, which keeps unit tests
fast and deterministic without downloading hundreds of megabytes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol, cast, runtime_checkable

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


@runtime_checkable
class _EmbeddingModel(Protocol):
    """Minimal surface of the sentence-transformers API we rely on."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        convert_to_numpy: bool = ...,
        show_progress_bar: bool = ...,
    ) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class LocalEmbedder:
    """Wraps a local sentence-transformers model with a small embedding API."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        model: _EmbeddingModel | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device if device is not None else settings.embedding_device
        self.batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else settings.embedding_normalize
        self._model = model
        self._dimension = dimension if isinstance(dimension, int) else None

    @property
    def model(self) -> _EmbeddingModel:
        """The underlying sentence-transformers model, loaded on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensionality of the embeddings produced by this model."""
        if self._dimension is None:
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None or dimension <= 0:
                raise EmbeddingError(
                    f"Model {self.model_name!r} reported an invalid embedding dimension."
                )
            self._dimension = dimension
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts into a ``list`` of float vectors.

        Batching is delegated to the model (``batch_size`` is honored), so callers
        may pass arbitrarily large lists without chunking by hand.
        """
        if not texts:
            return []
        if not all(isinstance(text, str) for text in texts):
            raise EmbeddingError("All inputs to embed_texts must be strings.")

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        rows: list[list[float]] = []
        for vector in vectors:
            try:
                rows.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    f"Unsupported embedding output from model {self.model_name!r}."
                ) from exc
        return rows

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into one float vector."""
        if not query or not query.strip():
            raise EmbeddingError("Query text must be a non-empty string.")
        return self.embed_texts([query])[0]

    def _load_model(self) -> _EmbeddingModel:
        """Import and instantiate the sentence-transformers model lazily."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; install the project "
                "dependencies to use local embeddings."
            ) from exc

        logger.info("Loading embedding model %s (device=%s)", self.model_name, self.device)
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise EmbeddingError(
                f"Failed to load embedding model {self.model_name!r}: {exc}"
            ) from exc
        # SentenceTransformer's overloaded ``encode`` is richer than our protocol,
        # so narrow it to the small surface we depend on.
        return cast(_EmbeddingModel, model)


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """Return a cached default :class:`LocalEmbedder` for dependency injection."""
    return LocalEmbedder()