"""Access-control tests: the public/members filter must hold **in every index**.

``tests/test_retrieval.py`` proves Chroma filters by visibility, and this file
does the same for the BM25 index, then checks that the Retriever passes the
right visibility set to *both* indexes in hybrid mode — including an end-to-end
run over a real in-memory Chroma collection and a real SQLite FTS5 index.

The rule under test: a public caller can never receive a ``members`` chunk,
whatever its rank, and the filtering never happens in the prompt.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest

from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index, IndexEntry
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
)
from mc_rag.rag.chunking import Chunk
from mc_rag.rag.retrieval import Retriever

PUBLIC_MEMBERS_TERM = "confidentiel-interne-xyz"


class FixedModel:
    """Stand-in embedding model returning one constant vector."""

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        sentences: Any,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        return np.array([[1.0, 0.0] for _ in sentences], dtype=float)


def fixed_embedder() -> LocalEmbedder:
    """A :class:`LocalEmbedder` whose query vector is always ``[1, 0]``."""
    return LocalEmbedder(model_name="fixed", model=FixedModel(), dimension=2)


def make_chunk(chunk_id: str, *, visibility: str, text: str) -> Chunk:
    """Chunk carrying the metadata shape produced by the chunker."""
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{visibility}",
        "source": f"data/raw/{visibility}/note.md",
        "format": "markdown",
        "visibility": visibility,
        "title": visibility.capitalize(),
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
def store() -> ChromaStore:
    """A fresh in-memory collection."""
    return ChromaStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"visibility_{uuid.uuid4().hex}",
        persist=False,
    )


@pytest.fixture
def index(tmp_path: Path) -> BM25Index:
    """A fresh file-backed BM25 index."""
    return BM25Index(tmp_path / "bm25.sqlite3")


def seed(store: ChromaStore, index: BM25Index) -> None:
    """Index one public and one members chunk, both an exact lexical match."""
    chunks = [
        make_chunk(
            "pub_strong",
            visibility=VISIBILITY_PUBLIC,
            text="Le bureau public annonce le hackathon ouvert a tous.",
        ),
        make_chunk(
            "mem_strong",
            visibility=VISIBILITY_MEMBERS,
            text=f"Memoire interne : {PUBLIC_MEMBERS_TERM} reste secret.",
        ),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [1.0, 0.0]])
    index.upsert(
        [
            IndexEntry(chunk_id=chunk.chunk_id, text=chunk.text, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]
    )


# --------------------------------------------------------------------------- #
# BM25 index
# --------------------------------------------------------------------------- #


def test_bm25_public_search_never_returns_members_chunks(index: BM25Index) -> None:
    index.upsert(
        [
            IndexEntry(
                chunk_id="public",
                text="budget public du semestre",
                metadata={"visibility": VISIBILITY_PUBLIC, "source": "public.md"},
            ),
            IndexEntry(
                chunk_id="members",
                text="budget public du semestre et salaires",
                metadata={"visibility": VISIBILITY_MEMBERS, "source": "members.md"},
            ),
        ]
    )

    hits = index.search("budget public", visibilities=(VISIBILITY_PUBLIC,), limit=10)

    assert [hit.chunk_id for hit in hits] == ["public"]


def test_bm25_members_only_content_is_invisible_to_public(index: BM25Index) -> None:
    index.upsert(
        [
            IndexEntry(
                chunk_id="members",
                text=f"Rappele : {PUBLIC_MEMBERS_TERM} est interdit de diffusion.",
                metadata={"visibility": VISIBILITY_MEMBERS, "source": "members.md"},
            )
        ]
    )

    # The members chunk is a perfect match, and still not returned: the filter
    # is part of the index query itself.
    assert index.search(PUBLIC_MEMBERS_TERM, visibilities=(VISIBILITY_PUBLIC,)) == []
    as_members = index.search(PUBLIC_MEMBERS_TERM, visibilities=ALL_VISIBILITIES)
    assert [hit.chunk_id for hit in as_members] == ["members"]


def test_bm25_member_search_sees_both_visibilities(index: BM25Index) -> None:
    index.upsert(
        [
            IndexEntry(
                chunk_id="public",
                text="hackathon public",
                metadata={"visibility": VISIBILITY_PUBLIC, "source": "a.md"},
            ),
            IndexEntry(
                chunk_id="members",
                text="hackathon prive",
                metadata={"visibility": VISIBILITY_MEMBERS, "source": "b.md"},
            ),
        ]
    )

    hits = index.search("hackathon", visibilities=ALL_VISIBILITIES, limit=10)

    assert {hit.chunk_id for hit in hits} == {"public", "members"}


# --------------------------------------------------------------------------- #
# Retriever: the allowed set is pushed into both indexes
# --------------------------------------------------------------------------- #


class SpyStore:
    """Chroma double recording the visibility set it was queried with."""

    def __init__(self) -> None:
        self.visibilities: tuple[str, ...] | None = None

    def search_by_text(
        self,
        text: str,
        embedder: Any,
        *,
        top_k: int | None = None,
        visibilities: Any = None,
        where: Any = None,
    ) -> list[Any]:
        self.visibilities = tuple(visibilities or ())
        return []


class SpyLexical:
    """BM25 double recording the visibility set it was queried with."""

    def __init__(self) -> None:
        self.visibilities: tuple[str, ...] | None = None

    def search(
        self, query: str, *, visibilities: Sequence[str], limit: int = 20
    ) -> list[Any]:
        self.visibilities = tuple(visibilities)
        return []


def test_retriever_sends_the_allowed_visibilities_to_both_indexes() -> None:
    store = SpyStore()
    lexical = SpyLexical()
    engine = Retriever(
        store=store,  # type: ignore[arg-type]
        embedder=fixed_embedder(),
        lexical=lexical,  # type: ignore[arg-type]
        hybrid_enabled=True,
    )

    engine.retrieve("question publique", is_member=False)
    assert store.visibilities == (VISIBILITY_PUBLIC,)
    assert lexical.visibilities == (VISIBILITY_PUBLIC,)

    engine.retrieve("question membre", is_member=True)
    assert store.visibilities == ALL_VISIBILITIES
    assert lexical.visibilities == ALL_VISIBILITIES


# --------------------------------------------------------------------------- #
# End to end: real Chroma + real BM25
# --------------------------------------------------------------------------- #


def hybrid_retriever(store: ChromaStore, index: BM25Index) -> Retriever:
    """A retriever exercising both real indexes with permissive thresholds."""
    return Retriever(
        store=store,
        embedder=fixed_embedder(),
        lexical=index,
        top_k=10,
        min_score=0.0,
        hybrid_enabled=True,
        hybrid_bm25_min_score=0.0,
    )


def test_hybrid_public_caller_never_receives_members_chunks(
    store: ChromaStore, index: BM25Index
) -> None:
    seed(store, index)

    result = hybrid_retriever(store, index).retrieve("hackathon secret", is_member=False)

    assert result.hybrid is True
    assert result.hits
    assert "mem_strong" not in {hit.chunk_id for hit in result.hits}
    assert all(hit.metadata.get("visibility") == VISIBILITY_PUBLIC for hit in result.hits)


def test_hybrid_members_only_term_yields_nothing_for_public(
    store: ChromaStore, index: BM25Index
) -> None:
    seed(store, index)

    result = hybrid_retriever(store, index).retrieve(
        f"informations {PUBLIC_MEMBERS_TERM}", is_member=False
    )

    assert "mem_strong" not in {hit.chunk_id for hit in result.hits}
    assert all(hit.metadata.get("visibility") == VISIBILITY_PUBLIC for hit in result.hits)


def test_hybrid_member_caller_sees_members_chunks(
    store: ChromaStore, index: BM25Index
) -> None:
    seed(store, index)

    result = hybrid_retriever(store, index).retrieve(
        f"informations {PUBLIC_MEMBERS_TERM}", is_member=True
    )

    ids = {hit.chunk_id for hit in result.hits}
    assert "mem_strong" in ids
    assert "pub_strong" in ids


def test_vector_only_mode_matches_the_same_isolation(
    store: ChromaStore, index: BM25Index
) -> None:
    seed(store, index)

    engine = Retriever(
        store=store,
        embedder=fixed_embedder(),
        lexical=index,
        top_k=10,
        min_score=0.0,
        hybrid_enabled=False,
    )
    result = engine.retrieve("hackathon secret", is_member=False)

    assert result.hybrid is False
    assert result.hits
    assert "mem_strong" not in {hit.chunk_id for hit in result.hits}
