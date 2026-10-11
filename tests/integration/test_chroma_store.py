"""Integration tests for :mod:`mc_rag.infrastructure.vector_store.chroma_store`.

These use a real in-memory Chroma instance (no model downloads) with synthetic
embeddings, so they verify insertion, upsert semantics, and — critically —
metadata-filtered (visibility) retrieval end to end.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import chromadb
import numpy as np
import pytest

from mc_rag.domain.exceptions import VectorStoreError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
    SearchResult,
    allowed_visibilities,
)
from mc_rag.rag.chunking import Chunk


class FixedModel:
    """Fake embedder that collapses every text to a fixed query vector."""

    def get_sentence_embedding_dimension(self) -> int | None:
        return 2

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        vectors = np.zeros((len(sentences), 2), dtype=float)
        for index in range(len(sentences)):
            vectors[index, :] = [1.0, 0.0]
        return vectors


def make_chunk(
    text: str,
    chunk_id: str,
    *,
    doc_id: str = "doc-1",
    visibility: str = VISIBILITY_PUBLIC,
    section_heading: str | None = "Overview",
) -> Chunk:
    """Build a Chunk carrying the metadata shape produced by the chunker."""
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": f"{doc_id}.md",
        "format": "markdown",
        "title": "Test Document",
        "visibility": visibility,
        "section_heading": section_heading,
        "section_anchor": "overview",
        "section_level": 1,
        "chunk_index": 0,
        "chunk_count": 1,
        "start_char": 0,
        "end_char": len(text),
        "document_metadata": {"source_dir": "public"},
    }
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        index=0,
        start_char=0,
        end_char=len(text),
        metadata=metadata,
    )


@pytest.fixture
def store() -> Iterator[ChromaStore]:
    # chromadb shares one in-process ephemeral system, so a unique collection
    # name per test keeps cases isolated from one another.
    client = chromadb.EphemeralClient()
    yield ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


def test_add_chunks_and_count(store: ChromaStore) -> None:
    chunks = [
        make_chunk("public chunk text", "c1"),
        make_chunk("members chunk text", "c2", visibility=VISIBILITY_MEMBERS),
    ]
    written = store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    assert written == 2
    assert store.count() == 2


def test_upsert_replaces_existing_chunks(store: ChromaStore) -> None:
    store.add_chunks([make_chunk("first version", "c1")], [[1.0, 0.0]])
    store.add_chunks([make_chunk("second version", "c1")], [[1.0, 0.0]])

    assert store.count() == 1
    (result,) = store.search([1.0, 0.0], visibilities=(VISIBILITY_PUBLIC,))
    assert result.text == "second version"


def test_mismatched_inputs_raise_value_error(store: ChromaStore) -> None:
    with pytest.raises(ValueError):
        store.add_chunks([make_chunk("a", "c1")], [[1.0, 0.0], [0.0, 1.0]])
    assert store.add_chunks([], []) == 0


def test_public_only_query_excludes_members(store: ChromaStore) -> None:
    store.add_chunks(
        [
            make_chunk("public chunk text", "pub", doc_id="d-public"),
            make_chunk(
                "members chunk text",
                "mem",
                doc_id="d-members",
                visibility=VISIBILITY_MEMBERS,
            ),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search([1.0, 0.0], top_k=5)

    assert [r.chunk_id for r in results] == ["pub"]
    assert results[0].metadata["visibility"] == VISIBILITY_PUBLIC


def test_member_query_sees_both_visibilities(store: ChromaStore) -> None:
    store.add_chunks(
        [
            make_chunk("public chunk text", "pub", doc_id="d-public"),
            make_chunk(
                "members chunk text",
                "mem",
                doc_id="d-members",
                visibility=VISIBILITY_MEMBERS,
            ),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search([1.0, 0.0], top_k=5, visibilities=ALL_VISIBILITIES)

    assert {r.chunk_id for r in results} == {"pub", "mem"}
    assert results[0].chunk_id == "pub"
    assert results[0].score > results[1].score


def test_extra_where_filter_combines_with_visibility(store: ChromaStore) -> None:
    store.add_chunks(
        [
            make_chunk("doc two chunk", "c-2a", doc_id="doc-2"),
            make_chunk("doc one chunk", "c-1a", doc_id="doc-1"),
        ],
        [[1.0, 0.0], [0.9, 0.1]],
    )

    results = store.search(
        [1.0, 0.0],
        top_k=5,
        visibilities=(VISIBILITY_PUBLIC, VISIBILITY_MEMBERS),
        where={"doc_id": "doc-2"},
    )

    assert [r.chunk_id for r in results] == ["c-2a"]


def test_search_by_text_uses_embedder(store: ChromaStore) -> None:
    store.add_chunks(
        [
            make_chunk("public chunk text", "pub"),
            make_chunk("members chunk text", "mem", visibility=VISIBILITY_MEMBERS),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    embedder = LocalEmbedder(model_name="fake-model", model=FixedModel())

    results = store.search_by_text("some query", embedder, visibilities=ALL_VISIBILITIES)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)


def test_empty_store_search_returns_no_results(store: ChromaStore) -> None:
    assert store.search([1.0, 0.0]) == []


def test_invalid_top_k_raises(store: ChromaStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=0)


def test_unknown_visibility_raises(store: ChromaStore) -> None:
    with pytest.raises(VectorStoreError):
        store.search([1.0, 0.0], visibilities=("secret",))


def test_metadata_is_sanitized_for_chroma(store: ChromaStore) -> None:
    store.add_chunks([make_chunk("some text", "c1")], [[1.0, 0.0]])

    raw = store.client.get_collection(store.collection_name).get(
        ids=["c1"], include=["metadatas"]
    )
    metadatas = raw["metadatas"]
    assert metadatas is not None
    stored = metadatas[0]
    assert isinstance(stored, dict)

    assert "section_heading" in stored
    assert "document_metadata" in stored
    raw_document_metadata = stored["document_metadata"]
    assert isinstance(raw_document_metadata, str)
    assert json.loads(raw_document_metadata) == {"source_dir": "public"}
    assert all(value is not None for value in stored.values())


def test_delete_document_removes_all_chunks(store: ChromaStore) -> None:
    store.add_chunks(
        [
            make_chunk("doc two chunk", "c-2a", doc_id="doc-2"),
            make_chunk("doc two chunk b", "c-2b", doc_id="doc-2"),
            make_chunk("doc one chunk", "c-1a", doc_id="doc-1"),
        ],
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    )
    assert store.count() == 3

    removed = store.delete_document("doc-2")

    assert removed == 2
    assert store.count() == 1


def test_allowed_visibilities_helper() -> None:
    assert allowed_visibilities(is_member=False) == (VISIBILITY_PUBLIC,)
    assert allowed_visibilities(is_member=True) == (
        VISIBILITY_PUBLIC,
        VISIBILITY_MEMBERS,
    )


def test_persistent_store_round_trips(tmp_path: Any) -> None:
    directory = str(tmp_path / "chroma")
    a = ChromaStore(
        collection_name="persist_corpus",
        persist_directory=directory,
        persist=True,
    )
    a.add_chunks([make_chunk("persisted text", "c1")], [[1.0, 0.0]])
    assert a.count() == 1

    b = ChromaStore(
        collection_name="persist_corpus",
        persist_directory=directory,
        persist=True,
    )
    assert b.count() == 1
    (result,) = b.search([1.0, 0.0], visibilities=(VISIBILITY_PUBLIC,))
    assert result.text == "persisted text"