"""Lexical (BM25) index backed by SQLite FTS5.

Why SQLite FTS5
---------------
SQLite ships with the project already (conversations, feedback), FTS5 provides
BM25 ranking natively (``bm25()``) and the index is persistent, so enabling or
rebuilding hybrid retrieval needs no extra service or dependency.

Two tables, kept in sync by this module:

``chunks``
    One row per chunk: identity, **visibility**, the original text and the
    metadata (JSON), plus ``content`` — the pre-normalized copy indexed by FTS5.
``chunks_fts``
    An *external content* FTS5 table (``content='chunks'``) built on
    ``content``. FTS5 does not maintain it by itself: every write here goes
    through :class:`BM25Index`, which deletes the FTS entry **before** the row.

Scores
------
``bm25()`` returns a **negative** number where *smaller is better*, and its
magnitude depends on the corpus (a one-document index scores ≈ ``-1e-6``). The
retriever therefore uses ``force = -bm25`` and compares it against
``HYBRID_BM25_MIN_SCORE`` instead of assuming a fixed range.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import LexicalIndexError
from mc_rag.infrastructure.lexical_store.tokenizer import build_match_expression, normalize
from mc_rag.infrastructure.vector_store.chroma_store import ALL_VISIBILITIES, VISIBILITY_PUBLIC

logger = logging.getLogger(__name__)

# ``content`` is indexed; identity/visibility/text/metadata stay in ``chunks``.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL,
    visibility TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_visibility ON chunks(visibility);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
"""

_METADATA_ERROR = "Lexical index query failed: {}"


@dataclass(slots=True, frozen=True)
class IndexEntry:
    """One chunk as stored by the lexical index."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def visibility(self) -> str:
        """The entry's visibility (defaults to ``public`` like Chroma)."""
        value = self.metadata.get("visibility")
        return str(value) if value else VISIBILITY_PUBLIC

    @property
    def doc_id(self) -> str:
        """The owning document id, derived from the source when absent."""
        value = self.metadata.get("doc_id")
        if value:
            return str(value)
        source = str(self.metadata.get("source") or "")
        if not source:
            return "unknown-doc"
        return hashlib.sha1(source.encode()).hexdigest()[:16]


@dataclass(slots=True, frozen=True)
class LexicalHit:
    """A search hit from the BM25 index."""

    entry: IndexEntry
    bm25_score: float = 0.0  # raw FTS5 value: <= 0, smaller is better

    @property
    def force(self) -> float:
        """``-bm25``: **higher** means a stronger lexical match."""
        return -self.bm25_score

    @property
    def chunk_id(self) -> str:
        """The hit's chunk id."""
        return self.entry.chunk_id


class BM25Index:
    """Persistent BM25 index; safe to share across threads."""

    def __init__(self, path: str | Path | None = None) -> None:
        settings = get_settings()
        self._path = Path(path) if path is not None else Path(settings.bm25_index_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """File backing the index (``:memory:`` when no file is used)."""
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        """Lazily opened connection with the schema applied."""
        if self._connection is None:
            with self._lock:
                if self._connection is not None:
                    return self._connection
                try:
                    if str(self._path) != ":memory:":
                        self._path.parent.mkdir(parents=True, exist_ok=True)
                    connection = sqlite3.connect(str(self._path), check_same_thread=False)
                    connection.executescript(_SCHEMA)
                    connection.commit()
                except sqlite3.Error as exc:
                    raise LexicalIndexError(
                        f"Unable to open the lexical index at {self._path}: {exc}"
                    ) from exc
                self._connection = connection
        return self._connection

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def upsert(self, entries: Sequence[IndexEntry]) -> int:
        """Insert or replace *entries*, keyed by ``chunk_id`` (idempotent).

        Returns:
            The number of entries written.
        """
        if not entries:
            return 0
        connection = self.connection
        try:
            with self._lock, connection:
                for entry in entries:
                    self._replace(connection, entry)
        except sqlite3.Error as exc:
            raise LexicalIndexError(_METADATA_ERROR.format(exc)) from exc
        return len(entries)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to *doc_id*. Returns the removal count."""
        connection = self.connection
        row_ids: list[int] = []
        try:
            with self._lock, connection:
                row_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
                    )
                ]
                self._delete_row_ids(connection, row_ids)
        except sqlite3.Error as exc:
            raise LexicalIndexError(_METADATA_ERROR.format(exc)) from exc
        if row_ids:
            logger.info("Removed %d chunks of %s from the lexical index", len(row_ids), doc_id)
        return len(row_ids)

    def clear(self) -> int:
        """Drop every entry (used before a full rebuild)."""
        connection = self.connection
        try:
            with self._lock, connection:
                count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                connection.execute("DELETE FROM chunks_fts")
                connection.execute("DELETE FROM chunks")
        except sqlite3.Error as exc:
            raise LexicalIndexError(_METADATA_ERROR.format(exc)) from exc
        return count

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        *,
        visibilities: Sequence[str],
        limit: int = 20,
    ) -> list[LexicalHit]:
        """Return the best BM25 hits for *query*, restricted to *visibilities*.

        The visibility restriction happens **inside** the index (SQL ``WHERE``),
        so a public caller can never see a members-only chunk, whatever its rank.

        Returns:
            Hits ordered by increasing ``bm25()`` (best first); empty when the
            query holds no token or the index is empty.
        """
        if limit <= 0:
            raise LexicalIndexError("limit must be a positive integer.")
        allowed = _validate_visibilities(visibilities)
        match = build_match_expression(query)
        if match is None:
            return []

        placeholders = ", ".join("?" for _ in allowed)
        sql = (
            "SELECT c.chunk_id, c.text, c.metadata, bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ? AND c.visibility IN ({placeholders}) "
            "ORDER BY score ASC, c.id ASC "
            "LIMIT ?"
        )
        connection = self.connection
        try:
            with self._lock:
                rows = connection.execute(sql, (match, *allowed, limit)).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError(_METADATA_ERROR.format(exc)) from exc

        hits: list[LexicalHit] = []
        for chunk_id, text, metadata, score in rows:
            hits.append(
                LexicalHit(
                    entry=IndexEntry(
                        chunk_id=str(chunk_id),
                        text=str(text),
                        metadata=_load_metadata(metadata),
                    ),
                    bm25_score=float(score),
                )
            )
        return hits

    def count(self) -> int:
        """Number of indexed chunks."""
        connection = self.connection
        try:
            with self._lock:
                return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        except sqlite3.Error as exc:
            raise LexicalIndexError(_METADATA_ERROR.format(exc)) from exc

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _replace(connection: sqlite3.Connection, entry: IndexEntry) -> None:
        """Replace the row for ``entry.chunk_id`` and refresh its FTS entry."""
        existing = connection.execute(
            "SELECT id FROM chunks WHERE chunk_id = ?", (entry.chunk_id,)
        ).fetchone()
        if existing is not None:
            row_id = int(existing[0])
            # The FTS entry must go first: it is keyed by the content rowid.
            connection.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row_id,))
            connection.execute("DELETE FROM chunks WHERE id = ?", (row_id,))

        cursor = connection.execute(
            "INSERT INTO chunks (chunk_id, doc_id, visibility, text, metadata, content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry.chunk_id,
                entry.doc_id,
                entry.visibility,
                entry.text,
                _dump_metadata(entry.metadata),
                normalize(entry.text),
            ),
        )
        inserted_id = cursor.lastrowid
        if inserted_id is None:
            raise LexicalIndexError(
                f"Unable to record chunk {entry.chunk_id!r} in the lexical index."
            )
        connection.execute(
            "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
            (int(inserted_id), normalize(entry.text)),
        )

    @staticmethod
    def _delete_row_ids(connection: sqlite3.Connection, row_ids: Sequence[int]) -> None:
        """Delete rows (FTS first) for *row_ids*."""
        for row_id in row_ids:
            connection.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row_id,))
            connection.execute("DELETE FROM chunks WHERE id = ?", (row_id,))


def _validate_visibilities(visibilities: Sequence[str]) -> tuple[str, ...]:
    """Return *visibilities* de-duplicated, rejecting unknown values."""
    unique = tuple(dict.fromkeys(visibilities))
    if not unique:
        raise LexicalIndexError("At least one visibility must be allowed.")
    unknown = sorted(set(unique) - set(ALL_VISIBILITIES))
    if unknown:
        raise LexicalIndexError(f"Unknown visibility values: {unknown}")
    return unique


def _dump_metadata(metadata: dict[str, Any]) -> str:
    """Serialize chunk metadata to JSON (Chroma stores scalars only)."""
    return json.dumps(metadata, ensure_ascii=False, default=str)


def _load_metadata(raw: Any) -> dict[str, Any]:
    """Parse stored metadata, tolerating a malformed row."""
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Discarding malformed lexical metadata: %.120s", raw)
        return {}
    return loaded if isinstance(loaded, dict) else {}
