"""Unit tests for :mod:`mc_rag.infrastructure.embeddings.local_embedder`.

The real sentence-transformers model is never loaded here; a stub model is
injected so the adapter contract can be verified fast and offline.
"""

from __future__ import annotations

import threading
import time
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
        self.seen: list[list[str]] = []

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
        self.seen.append(list(sentences))
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


def test_embed_query_applies_query_prefix() -> None:
    model = FakeModel(dimension=2)
    embedder = LocalEmbedder(
        model_name="fake-model", model=model, dimension=2, query_prefix="query: "
    )

    query_vector = embedder.embed_query("what is the club")

    assert len(query_vector) == 2
    assert model.seen[-1] == ["query: what is the club"]


def test_embed_documents_applies_doc_prefix() -> None:
    model = FakeModel(dimension=4)
    embedder = LocalEmbedder(
        model_name="fake-model", model=model, dimension=4, doc_prefix="passage: "
    )

    vectors = embedder.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert model.seen[-1] == ["passage: a", "passage: b"]


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
    assert embedder.device == settings.embedding_device


def test_get_embedder_returns_cached_instance() -> None:
    assert get_embedder() is get_embedder()
    get_embedder.cache_clear()


def test_warm_up_embeds_one_query() -> None:
    model = FakeModel(dimension=4)
    embedder = LocalEmbedder(model_name="fake-model", model=model, dimension=4)

    embedder.warm_up()

    assert model.seen == [["query: warm up"]]


def test_is_loaded_tracks_the_lazy_load(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = LocalEmbedder(model_name="fake-model", dimension=4)
    monkeypatch.setattr(embedder, "_load_model", lambda: FakeModel(dimension=4), raising=False)
    assert embedder.is_loaded is False
    _ = embedder.model
    assert embedder.is_loaded is True


def test_concurrent_first_access_loads_the_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup warm-up thread and the first ingestion thread share one load."""
    embedder = LocalEmbedder(model_name="fake-model", dimension=4)
    loads: list[str] = []
    barrier = threading.Barrier(4)

    def fake_load() -> FakeModel:
        loads.append("loaded")
        time.sleep(0.05)  # widen the race window
        return FakeModel(dimension=4)

    monkeypatch.setattr(embedder, "_load_model", fake_load, raising=False)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            assert embedder.model is not None
        except BaseException as exc:  # noqa: BLE001 - surfaced through the list
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert loads == ["loaded"]
    assert embedder.is_loaded is True