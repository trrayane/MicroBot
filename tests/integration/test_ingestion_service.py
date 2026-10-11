"""Integration tests for :mod:`mc_rag.application.ingestion_service`.

The service is exercised end to end against a real in-memory Chroma store with
a deterministic, dependency-free embedding model, so the full
parse -> chunk -> embed -> index flow is covered without downloading the real
sentence-transformers weights.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest

from mc_rag.application.ingestion_service import (
    SUPPORTED_EXTENSIONS,
    IngestionService,
    IngestionStatus,
    parse_document,
)
from mc_rag.domain.exceptions import UnsupportedFormatError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
)

_DIMENSION = 8


class HashingModel:
    """Deterministic stand-in for the sentence-transformers API surface."""

    def get_sentence_embedding_dimension(self) -> int | None:
        return _DIMENSION

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        vectors = np.zeros((len(sentences), _DIMENSION), dtype=float)
        for row, sentence in enumerate(sentences):
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            for column in range(_DIMENSION):
                vectors[row, column] = (digest[column] + 1) / 256.0
        return vectors


def write_markdown(path: Path, body: str) -> Path:
    """Create *path* (and its parents) containing *body*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def service(tmp_path: Path) -> IngestionService:
    store = ChromaStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"ingest_{uuid.uuid4().hex}",
    )
    embedder = LocalEmbedder(model=HashingModel(), dimension=_DIMENSION)
    return IngestionService(store=store, embedder=embedder, raw_data_dir=tmp_path)


def test_infer_visibility_from_folder(service: IngestionService, tmp_path: Path) -> None:
    public = write_markdown(tmp_path / "public" / "a.md", "# A\n\nBody.")
    members = write_markdown(tmp_path / "members" / "b.md", "# B\n\nBody.")
    loose = write_markdown(tmp_path / "loose.md", "# C\n\nBody.")

    assert service.infer_visibility(public, tmp_path) == VISIBILITY_PUBLIC
    assert service.infer_visibility(members, tmp_path) == VISIBILITY_MEMBERS
    assert service.infer_visibility(loose, tmp_path) == VISIBILITY_PUBLIC


def test_discover_files_ignores_unsupported(service: IngestionService, tmp_path: Path) -> None:
    write_markdown(tmp_path / "public" / "a.md", "# A\n\nBody.")
    write_markdown(tmp_path / "public" / "notes.txt", "ignored")
    write_markdown(tmp_path / "public" / ".gitkeep", "")

    discovered = service.discover_files(tmp_path)

    assert [path.name for path in discovered] == ["a.md"]


def test_ingest_directory_indexes_with_inherited_visibility(
    service: IngestionService, tmp_path: Path
) -> None:
    write_markdown(tmp_path / "public" / "public-doc.md", "# Public\n\nEveryone reads this.")
    write_markdown(tmp_path / "members" / "members-doc.md", "# Members\n\nSecret handshake.")

    report = service.ingest_directory(tmp_path)

    assert report.total_files == 2
    assert report.indexed == 2
    assert report.failed == 0
    assert report.chunks_embedded == service.store.count()

    by_name = {Path(result.path).name: result for result in report.results}
    public_id = by_name["public-doc.md"].doc_id
    members_id = by_name["members-doc.md"].doc_id
    assert public_id is not None and members_id is not None

    public_meta = service.store.get_document_metadata(public_id)
    members_meta = service.store.get_document_metadata(members_id)
    assert public_meta is not None and public_meta["visibility"] == VISIBILITY_PUBLIC
    assert members_meta is not None and members_meta["visibility"] == VISIBILITY_MEMBERS


def test_public_search_excludes_member_content(
    service: IngestionService, tmp_path: Path
) -> None:
    write_markdown(tmp_path / "public" / "public-doc.md", "# Public\n\nEveryone reads this.")
    write_markdown(tmp_path / "members" / "members-doc.md", "# Members\n\nSecret handshake.")
    service.ingest_directory(tmp_path)

    public_hits = service.store.search_by_text("Public", service.embedder)
    assert public_hits
    assert all(hit.metadata["visibility"] == VISIBILITY_PUBLIC for hit in public_hits)

    everything = service.store.search_by_text(
        "Members", service.embedder, top_k=10, visibilities=ALL_VISIBILITIES
    )
    assert {hit.metadata["visibility"] for hit in everything} == {
        VISIBILITY_PUBLIC,
        VISIBILITY_MEMBERS,
    }


def test_reingest_is_idempotent(service: IngestionService, tmp_path: Path) -> None:
    write_markdown(tmp_path / "public" / "a.md", "# A\n\nStable content for idempotency.")
    first = service.ingest_directory(tmp_path)
    assert first.indexed == 1
    count_after_first = service.store.count()

    second = service.ingest_directory(tmp_path)

    assert second.unchanged == 1
    assert second.chunks_embedded == 0
    assert service.store.count() == count_after_first


def test_modified_file_is_reindexed_without_stale_chunks(
    service: IngestionService, tmp_path: Path
) -> None:
    long_body = "# A\n\n" + ("Sentence number. " * 120)
    target = write_markdown(tmp_path / "public" / "a.md", long_body)
    first = service.ingest_directory(tmp_path)
    doc_id = first.results[0].doc_id
    count_before = service.store.count()
    assert count_before > 1

    write_markdown(target, "# A\n\nShort replacement body.")

    second = service.ingest_directory(tmp_path)

    assert second.updated == 1
    assert second.results[0].doc_id == doc_id
    assert service.store.count() == second.chunks_embedded < count_before


def test_explicit_visibility_override(service: IngestionService, tmp_path: Path) -> None:
    write_markdown(tmp_path / "public" / "a.md", "# A\n\nBody text.")

    report = service.ingest_directory(tmp_path, visibility=VISIBILITY_MEMBERS)

    assert report.results[0].visibility == VISIBILITY_MEMBERS
    doc_id = report.results[0].doc_id
    assert doc_id is not None
    meta = service.store.get_document_metadata(doc_id)
    assert meta is not None and meta["visibility"] == VISIBILITY_MEMBERS
    assert service.store.search_by_text("Body", service.embedder) == []


def test_force_reindexes_unchanged_file(service: IngestionService, tmp_path: Path) -> None:
    write_markdown(tmp_path / "public" / "a.md", "# A\n\nBody.")
    service.ingest_directory(tmp_path)

    forced = service.ingest_directory(tmp_path, force=True)

    assert forced.updated == 1
    assert forced.chunks_embedded == service.store.count()


def test_missing_file_reports_failure(service: IngestionService, tmp_path: Path) -> None:
    result = service.ingest_file(tmp_path / "public" / "missing.md")

    assert result.status is IngestionStatus.FAILED
    assert result.error


def test_unsupported_extension_reports_failure(
    service: IngestionService, tmp_path: Path
) -> None:
    path = write_markdown(tmp_path / "public" / "notes.txt", "plain text")

    result = service.ingest_file(path)

    assert result.status is IngestionStatus.FAILED
    assert "extension" in (result.error or "").lower()


def test_empty_file_reports_failure(service: IngestionService, tmp_path: Path) -> None:
    path = write_markdown(tmp_path / "public" / "empty.md", "")

    result = service.ingest_file(path)

    assert result.status is IngestionStatus.FAILED


def test_invalid_visibility_raises(service: IngestionService, tmp_path: Path) -> None:
    write_markdown(tmp_path / "public" / "a.md", "# A\n\nBody.")

    with pytest.raises(ValueError):
        service.ingest_directory(tmp_path, visibility="secret")


def test_parse_document_dispatches_by_extension(tmp_path: Path) -> None:
    path = write_markdown(tmp_path / "doc.md", "# Title\n\nHello world.")

    parsed = parse_document(path)

    assert parsed.format == "markdown"
    assert parsed.text.strip()


def test_parse_document_rejects_unknown_extension(tmp_path: Path) -> None:
    path = write_markdown(tmp_path / "doc.xyz", "data")

    with pytest.raises(UnsupportedFormatError):
        parse_document(path)


def test_supported_extensions_cover_expected_formats() -> None:
    assert {".md", ".markdown", ".pdf", ".docx", ".html", ".htm"} <= SUPPORTED_EXTENSIONS
