"""Ingestion service: parse -> chunk -> embed -> index raw corpus files.

The service ties the previously built pieces together into a single,
idempotent pipeline:

    file path -> parser (by extension) -> chunker -> embedder -> Chroma store

Idempotency
-----------
Every file is fingerprinted with a SHA-256 hash of its raw bytes. The hash and
the embedding model name are persisted on each chunk's metadata, so a re-run can
compare them to the indexed values via :meth:`ChromaStore.get_document_metadata`.
When they match (and the visibility is unchanged) the file is skipped entirely.
Changing ``EMBEDDING_MODEL`` therefore triggers a re-index automatically.

When a file *has* changed, the fresh chunks are upserted first and only then are
the obsolete chunks of the previous version deleted. A failure while embedding
or writing leaves the previous version of the document searchable.

Visibility
----------
Files are expected to live under a ``public/`` or ``members/`` folder beneath
the corpus root (``data/raw/public`` and ``data/raw/members`` by default). The
first path segment relative to the root determines the visibility
automatically; anything else falls back to ``default_visibility``. An explicit
``visibility`` argument always wins.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import DomainError, UnsupportedFormatError
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder, get_embedder
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index, IndexEntry
from mc_rag.infrastructure.parsers import (
    ParsedDocument,
    parse_csv,
    parse_docx,
    parse_markdown,
    parse_pdf,
    parse_web,
    parse_xlsx,
    resolve_document_path,
)
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_PUBLIC,
    ChromaStore,
)
from mc_rag.rag.chunking import Chunk, chunk_document

logger = logging.getLogger(__name__)

Parser = Callable[[str | Path], ParsedDocument]

#: ``(stage, chunks_done, chunks_total)`` progress hook. Stages are
#: ``parsing``/``chunking`` (no chunk counts yet), ``embedding`` (updated after
#: every embed batch, which is the slow part on CPU) and ``indexing``.
ProgressCallback = Callable[[str, int, int], None]

_HASH_BLOCK_SIZE = 1 << 16


def _parse_web_file(path: str | Path) -> ParsedDocument:
    """Adapt :func:`parse_web` to the ``str | Path`` parser signature."""
    return parse_web(str(path))


_EXTENSION_PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".docx": parse_docx,
    ".html": _parse_web_file,
    ".htm": _parse_web_file,
    ".xlsx": parse_xlsx,
    ".csv": parse_csv,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTENSION_PARSERS)


def parse_document(path: str | Path) -> ParsedDocument:
    """Parse *path* with the parser registered for its extension.

    Raises:
        UnsupportedFormatError: No parser is registered for the file extension.
    """
    suffix = Path(path).suffix.lower()
    parser = _EXTENSION_PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFormatError(
            f"No parser registered for extension {suffix or '<none>'!r}: {path}"
        )
    return parser(path)


class IngestionStatus(StrEnum):
    """Outcome of ingesting a single file."""

    INDEXED = "indexed"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(slots=True)
class FileIngestionResult:
    """Per-file ingestion outcome."""

    path: str
    status: IngestionStatus
    doc_id: str | None = None
    visibility: str | None = None
    chunks: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the file was processed without an error."""
        return self.status is not IngestionStatus.FAILED


@dataclass(slots=True)
class IngestionReport:
    """Aggregate result of ingesting a batch of files."""

    root: str
    results: list[FileIngestionResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def _count(self, status: IngestionStatus) -> int:
        return sum(1 for result in self.results if result.status is status)

    @property
    def total_files(self) -> int:
        """Number of files considered."""
        return len(self.results)

    @property
    def indexed(self) -> int:
        """Files newly added to the index."""
        return self._count(IngestionStatus.INDEXED)

    @property
    def updated(self) -> int:
        """Files re-indexed because their content changed."""
        return self._count(IngestionStatus.UPDATED)

    @property
    def unchanged(self) -> int:
        """Files skipped because their content was already indexed."""
        return self._count(IngestionStatus.UNCHANGED)

    @property
    def failed(self) -> int:
        """Files that could not be ingested."""
        return self._count(IngestionStatus.FAILED)

    @property
    def chunks_embedded(self) -> int:
        """Chunks embedded for newly indexed or updated files."""
        fresh = {IngestionStatus.INDEXED, IngestionStatus.UPDATED}
        return sum(result.chunks for result in self.results if result.status in fresh)

    @property
    def errors(self) -> list[FileIngestionResult]:
        """The failed results, for reporting."""
        return [result for result in self.results if result.status is IngestionStatus.FAILED]


class IngestionService:
    """Orchestrates parsing, chunking, embedding and indexing of corpus files."""

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        embedder: LocalEmbedder | None = None,
        lexical: BM25Index | None = None,
        raw_data_dir: str | Path | None = None,
        default_visibility: str = VISIBILITY_PUBLIC,
        parser: Parser = parse_document,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lexical = lexical
        settings = get_settings()
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else Path(
            settings.raw_data_dir
        )
        self.default_visibility = default_visibility
        self._parse = parser
        # Embedding is CPU-bound (seconds per chunk on a small machine): running
        # two jobs at once makes each of them ~2x slower and can load the model
        # twice. Jobs therefore run one at a time; the API still returns
        # immediately and reports progress through the status tracker.
        self._ingest_lock = threading.RLock()

    @property
    def store(self) -> ChromaStore:
        """The vector store, defaulting to the configured Chroma collection."""
        if self._store is None:
            self._store = ChromaStore()
        return self._store

    @property
    def embedder(self) -> LocalEmbedder:
        """The embedder, defaulting to the cached local model."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def lexical(self) -> BM25Index:
        """The BM25 index kept in sync with the vector store.

        Hybrid retrieval reads it, so ingestion writes to it as well; a failure
        there is logged but never fails the ingestion (the vector store is the
        source of truth and ``scripts/rebuild_index.py`` can repair the gap).
        """
        if self._lexical is None:
            self._lexical = BM25Index()
        return self._lexical

    def _sync_lexical(self, chunks: list[Chunk]) -> None:
        """Upsert freshly written chunks into the BM25 index."""
        if not chunks:
            return
        entries = [
            IndexEntry(chunk_id=chunk.chunk_id, text=chunk.text, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]
        try:
            self.lexical.upsert(entries)
        except (DomainError, OSError) as exc:
            logger.warning("Lexical index not updated for %d chunks: %s", len(chunks), exc)

    def _drop_lexical(self, doc_id: str) -> None:
        """Remove a deleted document from the BM25 index."""
        try:
            self.lexical.delete_document(doc_id)
        except (DomainError, OSError) as exc:
            logger.warning("Lexical index not cleaned for %s: %s", doc_id, exc)

    def discover_files(self, root: str | Path | None = None) -> list[Path]:
        """Return supported files under *root*, sorted for deterministic runs."""
        base = self._resolve_root(root)
        if not base.exists():
            raise DomainError(f"Corpus directory not found: {base}")
        if not base.is_dir():
            raise DomainError(f"Corpus path is not a directory: {base}")
        return sorted(
            (
                path
                for path in base.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.as_posix(),
        )

    def infer_visibility(self, path: str | Path, root: str | Path | None = None) -> str:
        """Infer visibility from the first path segment under *root*.

        ``<root>/public/...`` -> ``"public"`` and ``<root>/members/...`` ->
        ``"members"``; any other layout returns :attr:`default_visibility`.
        """
        base = self._resolve_root(root)
        try:
            relative = Path(path).resolve().relative_to(base.resolve())
        except ValueError:
            relative = Path(path)
        if relative.parts and relative.parts[0].lower() in ALL_VISIBILITIES:
            return relative.parts[0].lower()
        return self.default_visibility

    def ingest_directory(
        self,
        root: str | Path | None = None,
        *,
        visibility: str | None = None,
        force: bool = False,
    ) -> IngestionReport:
        """Discover and ingest every supported file under *root*."""
        base = self._resolve_root(root)
        return self.ingest_files(
            self.discover_files(base),
            root=base,
            visibility=visibility,
            force=force,
        )

    def ingest_files(
        self,
        paths: Iterable[str | Path],
        *,
        root: str | Path | None = None,
        visibility: str | None = None,
        force: bool = False,
    ) -> IngestionReport:
        """Ingest an explicit iterable of file paths into a single report."""
        base = self._resolve_root(root)
        if visibility is not None:
            self._validate_visibility(visibility)

        started = time.perf_counter()
        results = [
            self._ingest_file(path, base, visibility=visibility, force=force)
            for path in paths
        ]
        return IngestionReport(
            root=str(base),
            results=results,
            elapsed_seconds=time.perf_counter() - started,
        )

    def ingest_file(
        self,
        path: str | Path,
        *,
        root: str | Path | None = None,
        visibility: str | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> FileIngestionResult:
        """Ingest a single file and return its outcome (never raising for I/O)."""
        if visibility is not None:
            self._validate_visibility(visibility)
        return self._ingest_file(
            path,
            self._resolve_root(root),
            visibility=visibility,
            force=force,
            progress=progress,
        )

    def ingest_parsed_document(
        self,
        document: ParsedDocument,
        *,
        doc_id: str,
        visibility: str = VISIBILITY_PUBLIC,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> FileIngestionResult:
        """Chunk, embed and index an already-parsed document (no file on disk).

        Used for content that does not come from a corpus file, such as a URL
        fetched on the fly or a block of pasted text. ``doc_id`` is the caller's
        stable identity for the document; ``content_hash`` is derived from the
        text so re-ingesting identical content is skipped.
        """
        self._validate_visibility(visibility)
        content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        try:
            existing = self.store.get_document_metadata(doc_id)
            if not force and _is_unchanged(existing, content_hash, visibility):
                return FileIngestionResult(
                    path=document.source,
                    status=IngestionStatus.UNCHANGED,
                    doc_id=doc_id,
                    visibility=visibility,
                )

            with self._ingest_lock:
                _report(progress, "chunking", 0, 0)
                chunks = chunk_document(document, visibility=visibility)
                if not chunks:
                    if existing is not None:
                        self.store.delete_document(doc_id)
                        self._drop_lexical(doc_id)
                        return FileIngestionResult(
                            path=document.source,
                            status=IngestionStatus.UPDATED,
                            doc_id=doc_id,
                            visibility=visibility,
                        )
                    return FileIngestionResult(
                        path=document.source,
                        status=IngestionStatus.UNCHANGED,
                        doc_id=doc_id,
                        visibility=visibility,
                    )

                _stamp_chunks(chunks, doc_id, content_hash)
                embeddings = self._embed_chunks(
                    chunks, existing=existing, force=force, progress=progress
                )
                _report(progress, "indexing", len(chunks), len(chunks))
                self.store.add_chunks(chunks, embeddings)
                if existing is not None:
                    self.store.delete_stale_chunks(doc_id, {chunk.chunk_id for chunk in chunks})
                self._sync_lexical(chunks)

            status = IngestionStatus.UPDATED if existing is not None else IngestionStatus.INDEXED
            logger.info("Ingested %s (%d chunks, %s)", document.source, len(chunks), visibility)
            return FileIngestionResult(
                path=document.source,
                status=status,
                doc_id=doc_id,
                visibility=visibility,
                chunks=len(chunks),
            )
        except (DomainError, OSError) as exc:
            logger.warning("Failed to ingest %s: %s", document.source, exc)
            return FileIngestionResult(
                path=document.source, status=IngestionStatus.FAILED, error=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - a batch run must not abort
            logger.exception("Unexpected error ingesting %s", document.source)
            return FileIngestionResult(
                path=document.source,
                status=IngestionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def stable_doc_id(key: str) -> str:
        """Deterministic document id for a non-file source (URL, pasted text)."""
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def _ingest_file(
        self,
        path: str | Path,
        base: Path,
        *,
        visibility: str | None,
        force: bool,
        progress: ProgressCallback | None = None,
    ) -> FileIngestionResult:
        candidate = Path(path)
        try:
            resolved = resolve_document_path(candidate)
            effective_visibility = visibility or self.infer_visibility(resolved, base)
            doc_id = self._document_id(resolved, base)
            content_hash = _content_hash(resolved)

            existing = self.store.get_document_metadata(doc_id)
            if not force and _is_unchanged(existing, content_hash, effective_visibility):
                logger.debug("Unchanged, skipping: %s", resolved)
                return FileIngestionResult(
                    path=str(resolved),
                    status=IngestionStatus.UNCHANGED,
                    doc_id=doc_id,
                    visibility=effective_visibility,
                )

            with self._ingest_lock:
                _report(progress, "parsing", 0, 0)
                document = self._parse(resolved)
                _report(progress, "chunking", 0, 0)
                chunks = chunk_document(document, visibility=effective_visibility)

                if not chunks:
                    if existing is not None:
                        self.store.delete_document(doc_id)
                        self._drop_lexical(doc_id)
                        return FileIngestionResult(
                            path=str(resolved),
                            status=IngestionStatus.UPDATED,
                            doc_id=doc_id,
                            visibility=effective_visibility,
                        )
                    return FileIngestionResult(
                        path=str(resolved),
                        status=IngestionStatus.UNCHANGED,
                        doc_id=doc_id,
                        visibility=effective_visibility,
                    )

                _stamp_chunks(chunks, doc_id, content_hash)
                # embed_documents adds the "passage: " prefix required by e5 models.
                embeddings = self._embed_chunks(
                    chunks, existing=existing, force=force, progress=progress
                )

                # Upsert first, then drop obsolete chunks: a failure above or in the
                # upsert leaves the previous version of the document searchable.
                _report(progress, "indexing", len(chunks), len(chunks))
                self.store.add_chunks(chunks, embeddings)
                if existing is not None:
                    self.store.delete_stale_chunks(doc_id, {chunk.chunk_id for chunk in chunks})
                self._sync_lexical(chunks)

            status = IngestionStatus.UPDATED if existing is not None else IngestionStatus.INDEXED
            logger.info("Ingested %s (%d chunks, %s)", resolved, len(chunks), effective_visibility)
            return FileIngestionResult(
                path=str(resolved),
                status=status,
                doc_id=doc_id,
                visibility=effective_visibility,
                chunks=len(chunks),
            )
        except (DomainError, OSError) as exc:
            logger.warning("Failed to ingest %s: %s", candidate, exc)
            return FileIngestionResult(
                path=str(candidate),
                status=IngestionStatus.FAILED,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - a batch run must not abort
            logger.exception("Unexpected error ingesting %s", candidate)
            return FileIngestionResult(
                path=str(candidate),
                status=IngestionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _resolve_root(self, root: str | Path | None) -> Path:
        return Path(root) if root is not None else self.raw_data_dir

    def _embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        existing: dict[str, Any] | None,
        force: bool,
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed *chunks*, reusing stored vectors for unchanged chunk ids.

        Chunk ids are content-addressed, so re-ingesting an edited document
        keeps most ids stable; their vectors are read back from the vector store
        instead of being recomputed (embedding dominates ingestion time on CPU).
        Reuse is skipped when ``force`` is set or when the stored document was
        indexed with a different embedding model.

        Vectors are computed batch by batch so ``progress`` reports ``embedding``
        after every batch instead of staying silent for minutes.
        """
        total = len(chunks)
        _report(progress, "embedding", 0, total)
        if not chunks:
            return []

        cached: dict[str, list[float]] = {}
        if existing is not None and not force:
            if existing.get("embedding_model") == get_settings().embedding_model:
                cached = self.store.get_chunk_embeddings([chunk.chunk_id for chunk in chunks])

        missing = [chunk for chunk in chunks if chunk.chunk_id not in cached]
        if cached:
            logger.info(
                "Reusing %d/%d stored embeddings (%s)",
                len(cached),
                total,
                chunks[0].metadata.get("source", "unknown source"),
            )

        # ``getattr``: embedders are duck-typed (test doubles may not expose
        # ``batch_size``); 32 matches the configured default.
        batch_size = max(1, getattr(self.embedder, "batch_size", 32))
        vectors: list[list[float]] = []
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            # embed_documents adds the "passage: " prefix required by e5 models.
            vectors.extend(self.embedder.embed_documents([chunk.text for chunk in batch]))
            _report(progress, "embedding", len(cached) + len(vectors), total)

        merged: list[list[float]] = []
        iterator = iter(vectors)
        for chunk in chunks:
            vector = cached.get(chunk.chunk_id)
            merged.append(vector if vector is not None else next(iterator))
        return merged

    @staticmethod
    def _document_id(path: Path, base: Path) -> str:
        """Stable document id derived from the path relative to the corpus root."""
        try:
            key = path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            key = path.resolve().as_posix()
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _validate_visibility(visibility: str) -> None:
        if visibility not in ALL_VISIBILITIES:
            raise ValueError(
                f"Unknown visibility {visibility!r}; expected one of {ALL_VISIBILITIES}."
            )


def _content_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_unchanged(existing: dict[str, Any] | None, content_hash: str, visibility: str) -> bool:
    """Whether an already-indexed document matches the incoming file exactly.

    The embedding model is part of the comparison: switching ``EMBEDDING_MODEL``
    forces a re-index even when the file content did not change.
    """
    if existing is None:
        return False
    return (
        existing.get("content_hash") == content_hash
        and existing.get("visibility") == visibility
        and existing.get("embedding_model") == get_settings().embedding_model
    )


def _stamp_chunks(chunks: list[Chunk], doc_id: str, content_hash: str) -> None:
    """Attach the ingestion identity, hash and embedding model to every chunk."""
    model = get_settings().embedding_model
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["embedding_model"] = model


def _report(progress: ProgressCallback | None, stage: str, done: int, total: int) -> None:
    """Invoke the progress hook, never letting a bad hook fail the ingestion."""
    if progress is None:
        return
    try:
        progress(stage, done, total)
    except Exception:  # noqa: BLE001 - progress must stay best-effort
        logger.debug("Progress callback failed", exc_info=True)