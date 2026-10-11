"""Unit tests for :mod:`mc_rag.infrastructure.embeddings.local_embedder`.

The real sentence-transformers model is never loaded here; a stub model is
injected so the adapter contract can be verified fast and offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import EmbeddingError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder, get_embedder


class FakeModel:
    """Minimal stand-in for the sentence-transformers API surface we use."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.batch_sizes: list[int] = []
        self.normalize_flags: list[bool] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        self.batch_sizes.append(batch_size)
        self.normalize_flags.append(normalize_embeddings)
        vectors = np.zeros((len(sentences), self.dimension), dtype=float)
        for index in range(len(sentences)):
            vectors[index, :] = float(index + 1)
        return vectors


def test_embed_texts_returns_one_vector_per_text() -> None:
    model = FakeModel(dimension=4)
    embedder = LocalEmbedder(model_name="fake-model", model=model)

    vectors = embedder.embed_texts(["one", "two", "three"])

    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)
    assert vectors[0] == [1.0, 1.0, 1.0, 1.0]
    assert vectors[2] == [3.0, 3.0, 3.0, 3.0]
    assert model.batch_sizes == [embedder.batch_size]
    assert model.normalize_flags == [True]


def test_embed_texts_empty_input_is_noop() -> None:
    model = FakeModel()
    embedder = LocalEmbedder(model_name="fake-model", model=model)

    assert embedder.embed_texts([]) == []
    assert model.batch_sizes == []


def test_batch_size_is_configurable_and_forwarded() -> None:
    model = FakeModel()
    embedder = LocalEmbedder(model_name="fake-model", model=model, batch_size=17)

    assert embedder.batch_size == 17
    embedder.embed_texts(["a", "b"])
    assert model.batch_sizes == [17]


def test_embed_query_matches_embed_texts() -> None:
    model = FakeModel(dimension=2)
    embedder = LocalEmbedder(model_name="fake-model", model=model)

    query_vector = embedder.embed_query("what is the club")
    text_vectors = embedder.embed_texts(["what is the club"])

    assert query_vector == text_vectors[0]
    assert len(query_vector) == 2


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_embed_query_rejects_blank_text(query: str) -> None:
    embedder = LocalEmbedder(model_name="fake-model", model=FakeModel())
    with pytest.raises(EmbeddingError):
        embedder.embed_query(query)


def test_non_string_input_is_rejected_with_custom_exception() -> None:
    embedder = LocalEmbedder(model_name="fake-model", model=FakeModel())
    # The adapter accepts Sequence[str] on paper but validates defensively.
    mixed: Sequence[Any] = ["ok", 5]
    with pytest.raises(EmbeddingError):
        embedder.embed_texts(mixed)


def test_dimension_comes_from_model() -> None:
    embedder = LocalEmbedder(model_name="fake-model", model=FakeModel(dimension=7))
    assert embedder.dimension == 7


def test_dimension_can_be_injected_without_a_model() -> None:
    embedder = LocalEmbedder(model_name="fake-model", dimension=16)
    assert embedder.dimension == 16


def test_defaults_reflect_settings() -> None:
    settings = get_settings()
    embedder = LocalEmbedder(model=FakeModel())

    assert embedder.model_name == settings.embedding_model
    assert embedder.batch_size == settings.embedding_batch_size
    assert embedder.normalize == settings.embedding_normalize
    assert embedder.device is None


def test_get_embedder_returns_cached_instance() -> None:
    assert get_embedder() is get_embedder()
    get_embedder.cache_clear()