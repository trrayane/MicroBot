"""Real-stack embedding round-trip test (guarded).

This test downloads the actual sentence-transformers model, which is large and
network dependent, so it is skipped unless ``MC_RAG_RUN_EMBEDDING_TESTS=1`` is
explicitly set. Everything else in the suite avoids the download by injecting
stub models.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import chromadb
import pytest

from mc_rag.config import get_settings
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.vector_store.chroma_store import (
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
)
from mc_rag.rag.chunking import Chunk

_RUN = os.getenv("MC_RAG_RUN_EMBEDDING_TESTS") == "1"


def make_chunk(text: str, chunk_id: str, *, visibility: str) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        index=0,
        start_char=0,
        end_char=len(text),
        metadata={
            "chunk_id": chunk_id,
            "doc_id": "real-doc",
            "source": "real-doc.md",
            "format": "markdown",
            "title": "Real Document",
            "visibility": visibility,
            "chunk_index": 0,
            "chunk_count": 1,
        },
    )


@pytest.mark.skipif(not _RUN, reason="set MC_RAG_RUN_EMBEDDING_TESTS=1 to run model-backed tests")
def test_real_embedding_roundtrip(tmp_path: Any) -> None:
    settings = get_settings()
    embedder = LocalEmbedder()
    assert embedder.dimension == settings.embedding_dimension

    chunks = [
        make_chunk("A cat sat on the mat and purred loudly.", "cat", visibility=VISIBILITY_PUBLIC),
        make_chunk(
            "Quantum chromodynamics describes the strong force between quarks.",
            "quark",
            visibility=VISIBILITY_MEMBERS,
        ),
    ]
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    store = ChromaStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"embed_roundtrip_{uuid.uuid4().hex}",
    )
    store.add_chunks(chunks, vectors)

    results = store.search_by_text(
        "A cat sleeping on a mat",
        embedder,
        top_k=5,
        visibilities=(VISIBILITY_PUBLIC, VISIBILITY_MEMBERS),
    )

    assert results[0].chunk_id == "cat"

    public_only = store.search_by_text("A cat sleeping on a mat", embedder, top_k=5)
    assert [result.chunk_id for result in public_only] == ["cat"]