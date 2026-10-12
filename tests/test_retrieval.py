"""Retrieval engine and visibility-filter security tests.

These tests exercise the two security-relevant responsibilities of the
retrieval stage:

* **Role-based access control** - a ``public`` caller may never receive a
  ``members`` chunk, and that filtering happens *inside* the vector store
  (via Chroma's ``$in`` clause), not as an afterthought in the prompt.
* **Relevance gating** - ``top_k`` and ``RETRIEVAL_MIN_SCORE`` bound what the
  pipeline is allowed to ground on.

A real in-memory Chroma collection is used with a deterministic fake embedder
so cosine scores are known exactly; no model download or network is required.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import numpy as np
import pytest

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import VectorStoreError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
    _visibility_clause,
    allowed_visibilities,
)
from mc_rag.rag.chunking import Chunk
from mc_rag.rag.retrieval import Retriever, build_search_query

# ==========================================================================================
# Fake embedders
# ==========================================================================================


class FixedModel:
    """Sentence-transformers stand-in that returns one constant vector."""

    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self._vector = vector

    def get_sentence_embedding_dimension(self) -> int:
        return len(self._vector)

    def encode(
        self,
        sentences: Any,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        return np.array([self._vector for _ in sentences], dtype=float)


class RecordingModel(FixedModel):
    """Fixed model that remembers the exact strings it was asked to encode."""

    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        super().__init__(vector)
        self.seen: list[str] = []

    def encode(self, sentences: Any, **kwargs: Any) -> Any:
        self.seen.extend(str(sentence) for sentence in sentences)
        return super().encode(sentences)


def fixed_embedder(vector: tuple[float, ...] = (1.0, 0.0)) -> LocalEmbedder:
    """A ``LocalEmbedder`` whose query vector is exactly *vector*."""
    return LocalEmbedder(model_name="fixed", model=FixedModel(vector), dimension=len(vector))


# ==========================================================================================
# Fixtures and helpers
# ==========================================================================================


@pytest.fixture
def store() -> ChromaStore:
    """A fresh, isolated in-memory collection (ephemeral client is global)."""
    return ChromaStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"retrieval_{uuid.uuid4().hex}",
        persist=False,
    )


def make_chunk(
    chunk_id: str,
    *,
    visibility: str,
    source: str,
    text: str | None = None,
) -> Chunk:
    body = text if text is not None else f"chunk {chunk_id}"
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "source": source,
        "format": "markdown",
        "visibility": visibility,
    }
    return Chunk(
        text=body,
        chunk_id=chunk_id,
        index=0,
        start_char=0,
        end_char=len(body),
        metadata=metadata,
    )


def seed(store: ChromaStore) -> None:
    """Index three public chunks and one members chunk with known cosine scores.

    With a query vector of ``[1, 0]`` the resulting similarity scores are:

    * ``pub_strong`` / ``mem_strong`` -> 1.0
    * ``pub_mid``                     -> ~0.707
    * ``pub_weak``                    -> 0.0
    """
    store.add_chunks(
        [
            make_chunk("pub_strong", visibility=VISIBILITY_PUBLIC, source="data/raw/public/a.md"),
            make_chunk("pub_mid", visibility=VISIBILITY_PUBLIC, source="data/raw/public/b.md"),
            make_chunk("pub_weak", visibility=VISIBILITY_PUBLIC, source="data/raw/public/c.md"),
            make_chunk("mem_strong", visibility=VISIBILITY_MEMBERS, source="data/raw/members/d.md"),
        ],
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    )


def retriever(
    store: ChromaStore,
    *,
    min_score: float = 0.0,
    top_k: int | None = None,
) -> Retriever:
    return Retriever(store=store, embedder=fixed_embedder(), min_score=min_score, top_k=top_k)


# ==========================================================================================
# Query vectorization: e5 "query: " prefix
# ==========================================================================================


def test_query_embedding_uses_query_prefix() -> None:
    model = RecordingModel()
    embedder = LocalEmbedder(
        model_name="recording", model=model, dimension=2, query_prefix="query: "
    )

    embedder.embed_query("bonjour le club")

    assert model.seen == ["query: bonjour le club"]


def test_configured_query_prefix_is_applied_by_default() -> None:
    settings = get_settings()
    model = RecordingModel()
    embedder = LocalEmbedder(model_name="recording", model=model, dimension=2)

    embedder.embed_query("salam")

    assert model.seen == [f"{settings.embedding_query_prefix}salam"]


# ==========================================================================================
# Visibility helpers and the $in clause
# ==========================================================================================


def test_public_caller_is_restricted_to_public() -> None:
    assert allowed_visibilities(is_member=False) == (VISIBILITY_PUBLIC,)


def test_member_sees_public_and_members() -> None:
    assert allowed_visibilities(is_member=True) == ALL_VISIBILITIES


def test_members_clause_uses_in_operator() -> None:
    clause = _visibility_clause(allowed_visibilities(is_member=True))
    assert clause == {"visibility": {"$in": ["public", "members"]}}


def test_public_clause_uses_equality() -> None:
    assert _visibility_clause(allowed_visibilities(is_member=False)) == {"visibility": "public"}


# ==========================================================================================
# Access control through the Retriever
# ==========================================================================================


def test_public_retrieval_never_returns_members_chunks(store: ChromaStore) -> None:
    seed(store)

    result = retriever(store, min_score=0.0, top_k=10).retrieve("anything", is_member=False)

    assert result.has_answer
    assert {hit.metadata["visibility"] for hit in result.hits} == {VISIBILITY_PUBLIC}
    assert "mem_strong" not in {hit.chunk_id for hit in result.hits}
    assert all("members" not in hit.metadata["source"] for hit in result.hits)


def test_public_top_match_cannot_be_a_members_chunk(store: ChromaStore) -> None:
    seed(store)

    # mem_strong ties for the best score, so a post-filter bug would surface it.
    result = retriever(store, min_score=0.0, top_k=1).retrieve("anything", is_member=False)

    assert len(result.hits) == 1
    assert result.hits[0].metadata["visibility"] == VISIBILITY_PUBLIC
    assert result.hits[0].chunk_id != "mem_strong"


def test_member_retrieval_includes_members_chunks(store: ChromaStore) -> None:
    seed(store)

    result = retriever(store, min_score=0.0, top_k=10).retrieve("anything", is_member=True)

    ids = {hit.chunk_id for hit in result.hits}
    assert "mem_strong" in ids
    assert {"pub_strong", "pub_mid", "pub_weak"} <= ids


def test_extra_where_cannot_widen_access(store: ChromaStore) -> None:
    seed(store)

    hits = store.search(
        fixed_embedder().embed_query("anything"),
        top_k=10,
        visibilities=(VISIBILITY_PUBLIC,),
        where={"visibility": VISIBILITY_MEMBERS},
    )

    assert hits == []


# ==========================================================================================
# Threshold and top_k gating
# ==========================================================================================


def test_min_score_drops_low_relevance_hits(store: ChromaStore) -> None:
    seed(store)

    result = retriever(store, min_score=0.8, top_k=10).retrieve("anything", is_member=True)

    ids = {hit.chunk_id for hit in result.hits}
    assert ids == {"pub_strong", "mem_strong"}
    assert all(hit.score >= 0.8 for hit in result.hits)
    # pub_mid (~0.707) is relevant but below the bar.
    assert "pub_mid" not in ids


def test_no_hits_when_everything_is_below_threshold(store: ChromaStore) -> None:
    seed(store)

    result = retriever(store, min_score=1.5, top_k=10).retrieve("anything", is_member=True)

    assert result.hits == []
    assert result.has_answer is False
    # best_score still reports the raw top score for observability.
    assert result.best_score == pytest.approx(1.0, abs=1e-6)


def test_top_k_bounds_the_number_of_hits(store: ChromaStore) -> None:
    seed(store)

    result = retriever(store, min_score=0.0, top_k=2).retrieve("anything", is_member=True)

    assert len(result.hits) == 2


def test_threshold_and_visibility_combine(store: ChromaStore) -> None:
    seed(store)

    public = retriever(store, min_score=0.8, top_k=10).retrieve("anything", is_member=False)
    member = retriever(store, min_score=0.8, top_k=10).retrieve("anything", is_member=True)

    public_ids = {hit.chunk_id for hit in public.hits}
    member_ids = {hit.chunk_id for hit in member.hits}
    assert public_ids == {"pub_strong"}
    assert member_ids == {"pub_strong", "mem_strong"}
    assert public_ids < member_ids


def test_retriever_defaults_come_from_settings(store: ChromaStore) -> None:
    settings = get_settings()

    engine = Retriever(store=store, embedder=fixed_embedder())

    assert engine.top_k == settings.retrieval_top_k
    assert engine.min_score == settings.retrieval_min_score


def test_search_by_text_rejects_blank_query(store: ChromaStore) -> None:
    with pytest.raises(VectorStoreError):
        store.search_by_text("   ", fixed_embedder())


# ==========================================================================================
# Search-query construction (follow-up handling)
# ==========================================================================================


def test_short_message_inherits_previous_user_context() -> None:
    history = [
        {"role": "user", "content": "Quand est le hackathon ?"},
        {"role": "assistant", "content": "En mai."},
    ]

    assert build_search_query("et où ?", history) == "Quand est le hackathon ? et où ?"


def test_long_message_is_used_verbatim() -> None:
    message = "Comment rejoindre le club de robotique de mon universite cette annee ?"

    assert build_search_query(message, [{"role": "user", "content": "salut"}]) == message


def test_short_message_without_user_turn_is_used_as_is() -> None:
    history = [{"role": "assistant", "content": "Bienvenue au club !"}]

    assert build_search_query("et où ?", history) == "et où ?"
