"""Chroma vector store adapter.

Wraps a Chroma collection (persistent on-disk or ephemeral/in-memory) used as
the RAG corpus. Chunks are upserted with their embeddings, full text and a
curated set of metadata, then retrieved via cosine similarity search.

Visibility model
----------------
Every chunk is tagged ``visibility`` = ``"public"`` or ``"members"``. Searches
carry an explicit set of *allowed* visibilities; the resulting Chroma ``where``
clause is combined with any caller-supplied filter using ``$and``, so a caller
can never widen its own access through an extra ``where``. Unauthenticated
callers should use the default of ``("public",)`` (see :func:`allowed_visibilities`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata, Where

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import VectorStoreError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.rag.chunking import Chunk

logger = logging.getLogger(__name__)

VISIBILITY_PUBLIC = "public"
VISIBILITY_MEMBERS = "members"
ALL_VISIBILITIES: tuple[str, ...] = (VISIBILITY_PUBLIC, VISIBILITY_MEMBERS)


def allowed_visibilities(*, is_member: bool) -> tuple[str, ...]:
    """Return the visibilities a caller is permitted to query.

    Members may see everything; everyone else is restricted to public content.
    """
    return ALL_VISIBILITIES if is_member else (VISIBILITY_PUBLIC,)


@dataclass(slots=True)
class SearchResult:
    """A single hit returned from a vector search."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    score: float = 0.0


class ChromaStore:
    """Thin, configurable adapter over a Chroma collection."""

    def __init__(
        self,
        *,
        client: ClientAPI | None = None,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        persist: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.chroma_collection
        self._persist_directory = (
            settings.chroma_persist_dir if persist_directory is None else persist_directory
        )
        self._persist = settings.chroma_persist if persist is None else persist
        self._client = client
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> ClientAPI:
        """The underlying Chroma client, created lazily on first access."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """The corpus collection, created/retrieved lazily on first access."""
        return self._resolve_collection()

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Upsert *chunks* with their precomputed *embeddings*.

        Uses upsert semantics so re-ingesting a document (same content-derived
        chunk ids) replaces existing records instead of duplicating them.

        Returns:
            The number of chunks written.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            return 0

        collection = self._resolve_collection()
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas: list[Metadata] = [_chunk_to_metadata(chunk) for chunk in chunks]
        vector_rows = np.asarray([list(embedding) for embedding in embeddings], dtype=np.float32)

        try:
            collection.upsert(
                ids=ids,
                embeddings=vector_rows,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to upsert {len(chunks)} chunks: {exc}") from exc
        return len(chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int | None = None,
        visibilities: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the corpus by vector, restricted to *visibilities*.

        Args:
            query_embedding: The embedding of the query text.
            top_k: Maximum number of results; defaults to ``retrieval_top_k``.
            visibilities: Visibilities the caller may see. Defaults to public-only.
            where: Additional metadata filter, ``AND``-combined with the
                visibility restriction (can never widen access).

        Returns:
            Search hits ordered by increasing cosine distance.
        """
        settings = get_settings()
        limit = settings.retrieval_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be a positive integer.")

        allowed = tuple(visibilities) if visibilities is not None else (VISIBILITY_PUBLIC,)
        where_filter = _combine_where(_visibility_clause(allowed), where)
        collection = self._resolve_collection()

        try:
            raw = collection.query(
                query_embeddings=np.asarray([list(query_embedding)], dtype=np.float32),
                n_results=limit,
                where=where_filter,
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        return _parse_query_result(raw)

    def search_by_text(
        self,
        text: str,
        embedder: LocalEmbedder,
        *,
        top_k: int | None = None,
        visibilities: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed *text* (with the query prefix) and search."""
        if not text or not text.strip():
            raise VectorStoreError("Search text must be a non-empty string.")
        query_embedding = embedder.embed_query(text)
        return self.search(
            query_embedding,
            top_k=top_k,
            visibilities=visibilities,
            where=where,
        )

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        try:
            return int(self._resolve_collection().count())
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to count collection: {exc}") from exc

    def list_documents(self) -> list[dict[str, Any]]:
        """Return one summary row per indexed document, sorted by title.

        Aggregates the stored chunk metadata by ``doc_id``. Each row has the
        ``doc_id``, ``title``, ``source``, ``visibility``, ``embedding_model``
        and ``chunks`` (number of chunks) keys.
        """
        try:
            result = self._resolve_collection().get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to list documents: {exc}") from exc

        documents: dict[str, dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            doc_id = str(meta.get("doc_id") or "unknown-doc")
            row = documents.get(doc_id)
            if row is None:
                source = str(meta.get("source") or "")
                row = {
                    "doc_id": doc_id,
                    "title": str(meta.get("title") or source or doc_id),
                    "source": source,
                    "visibility": str(meta.get("visibility") or VISIBILITY_PUBLIC),
                    "embedding_model": str(meta.get("embedding_model") or ""),
                    "chunks": 0,
                }
                documents[doc_id] = row
            row["chunks"] += 1
        return sorted(documents.values(), key=lambda row: str(row["title"]).lower())

    def stats(self) -> dict[str, Any]:
        """Return high-level counts for the collection."""
        documents = self.list_documents()
        return {
            "collection": self.collection_name,
            "documents": len(documents),
            "chunks": sum(int(row["chunks"]) for row in documents),
        }

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Return the metadata of a single chunk for *doc_id*, or ``None``.

        Used by the ingestion pipeline to detect unchanged documents from their
        stored ``content_hash`` and ``embedding_model`` without re-embedding.
        """
        try:
            result = self._resolve_collection().get(
                where=cast("Where", {"doc_id": doc_id}),
                limit=1,
                include=["metadatas"],
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to read document {doc_id!r}: {exc}") from exc
        metadatas = result.get("metadatas")
        if not metadatas:
            return None
        return dict(metadatas[0])

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk belonging to *doc_id*; returns rows removed."""
        ids = self._chunk_ids_for(doc_id)
        if not ids:
            return 0
        try:
            self._resolve_collection().delete(ids=ids)
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to delete document {doc_id!r}: {exc}") from exc
        return len(ids)

    def delete_stale_chunks(self, doc_id: str, keep_ids: Collection[str]) -> int:
        """Delete chunks of *doc_id* whose id is not in *keep_ids*.

        Called after a successful upsert when a document is re-ingested, so the
        old version stays searchable until the new one is fully written.
        """
        keep = set(keep_ids)
        stale = [chunk_id for chunk_id in self._chunk_ids_for(doc_id) if chunk_id not in keep]
        if not stale:
            return 0
        try:
            self._resolve_collection().delete(ids=stale)
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(
                f"Failed to delete stale chunks of {doc_id!r}: {exc}"
            ) from exc
        return len(stale)

    def _chunk_ids_for(self, doc_id: str) -> list[str]:
        """Return the ids of every stored chunk of *doc_id*."""
        try:
            result = self._resolve_collection().get(
                where=cast("Where", {"doc_id": doc_id}),
                include=[],
            )
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to list chunks of {doc_id!r}: {exc}") from exc
        return [str(chunk_id) for chunk_id in (result.get("ids") or [])]

    def _create_client(self) -> ClientAPI:
        """Build a persistent or ephemeral Chroma client from configuration."""
        try:
            if self._persist:
                Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
                return chromadb.PersistentClient(path=self._persist_directory)
            return chromadb.EphemeralClient()
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            raise VectorStoreError(f"Failed to initialize Chroma client: {exc}") from exc

    def _resolve_collection(self) -> chromadb.Collection:
        """Retrieve or create the corpus collection (cosine space)."""
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # noqa: BLE001 - normalize third-party errors
                raise VectorStoreError(
                    f"Failed to resolve Chroma collection {self.collection_name!r}: {exc}"
                ) from exc
        return self._collection


def _visibility_clause(visibilities: tuple[str, ...]) -> Where:
    """Build a Chroma ``where`` clause matching exactly *visibilities*."""
    unique = tuple(dict.fromkeys(visibilities))
    if not unique:
        raise VectorStoreError("At least one visibility must be allowed.")
    unknown = sorted(set(unique) - set(ALL_VISIBILITIES))
    if unknown:
        raise VectorStoreError(f"Unknown visibility values: {unknown}")
    if len(unique) == 1:
        return cast("Where", {"visibility": unique[0]})
    return cast("Where", {"visibility": {"$in": list(unique)}})


def _combine_where(*clauses: Where | dict[str, Any] | None) -> Where | None:
    """AND-combine present clauses; returns ``None`` when there are none."""
    present = [clause for clause in clauses if clause]
    if not present:
        return None
    if len(present) == 1:
        return cast("Where", present[0])
    return cast("Where", {"$and": present})


def _chunk_to_metadata(chunk: Chunk) -> Metadata:
    """Flatten ``Chunk`` metadata to Chroma-compatible scalar values.

    ``None`` values are dropped (Chroma rejects them) and nested dictionaries
    are JSON-encoded so the original structure is preserved for inspection.
    """
    meta = chunk.metadata
    source = str(meta.get("source") or "")
    result: dict[str, str | int | float | bool] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": str(meta.get("doc_id") or _derive_doc_id(source)),
        "source": source,
        "format": str(meta.get("format") or ""),
        "visibility": str(meta.get("visibility") or VISIBILITY_PUBLIC),
        "chunk_index": _coerce_int(meta.get("chunk_index"), chunk.index),
        "chunk_count": _coerce_int(meta.get("chunk_count"), 1),
        "start_char": _coerce_int(meta.get("start_char"), chunk.start_char),
        "end_char": _coerce_int(meta.get("end_char"), chunk.end_char),
    }

    for key in ("title", "section_heading", "section_anchor"):
        value = meta.get(key)
        if value:
            result[key] = str(value)

    section_level = meta.get("section_level")
    if section_level is not None:
        result["section_level"] = _coerce_int(section_level, 0)

    content_hash = meta.get("content_hash")
    if content_hash:
        result["content_hash"] = str(content_hash)

    embedding_model = meta.get("embedding_model")
    if embedding_model:
        result["embedding_model"] = str(embedding_model)

    nested = meta.get("document_metadata")
    if isinstance(nested, dict):
        result["document_metadata"] = json.dumps(nested, ensure_ascii=False, default=str)

    return result


def _coerce_int(value: Any, default: int) -> int:
    """Coerce *value* to an ``int``, falling back to *default*."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _derive_doc_id(source: str) -> str:
    """Stable document id derived from the source path when none is provided."""
    if not source:
        return "unknown-doc"
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def _parse_query_result(raw: Any) -> list[SearchResult]:
    """Extract typed :class:`SearchResult` rows from a raw Chroma response."""
    ids = raw.get("ids") or [[]]
    documents = raw.get("documents") or [[]]
    metadatas = raw.get("metadatas") or [[]]
    distances = raw.get("distances") or [[]]

    results: list[SearchResult] = []
    for index in range(len(ids[0])):
        chunk_id = str(ids[0][index])
        if not chunk_id:
            continue

        text = ""
        if index < len(documents[0]):
            document = documents[0][index]
            text = str(document) if document is not None else ""

        metadata: dict[str, Any] = {}
        if index < len(metadatas[0]):
            row = metadatas[0][index]
            if row:
                metadata = dict(row)

        distance = 0.0
        if index < len(distances[0]):
            distance = float(distances[0][index])

        results.append(
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                distance=distance,
                score=round(1.0 - distance, 6),
            )
        )
    return results