"""SQLite persistence for Discord users and document ownership."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class UserStore:
    """Thread-safe SQLite store for Discord users and document ownership."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        # Reentrant: the write methods hold the lock and then call the
        # ``_connection`` property, which takes the same lock to open the
        # connection on first use. A plain ``Lock`` self-deadlocked here, which
        # froze the very first ``POST /documents`` (or Discord login) forever.
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(self._path), check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.executescript(_SCHEMA)
                    conn.commit()
                    self._conn = conn
        return self._conn

    # -- users ---------------------------------------------------------------

    def upsert_user(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
        avatar: str | None,
        is_admin: bool,
    ) -> None:
        """Insert or update a Discord user."""
        with self._lock:
            conn = self._connection
            conn.execute(
                """
                INSERT INTO users (user_id, username, display_name, avatar, is_admin, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username=excluded.username,
                    display_name=excluded.display_name,
                    avatar=excluded.avatar,
                    is_admin=excluded.is_admin,
                    last_login=excluded.last_login
                """,
                (user_id, username, display_name, avatar, int(is_admin), time.time()),
            )
            conn.commit()

    def get_user(self, user_id: str) -> dict[str, object] | None:
        """Return a user row as a dict, or ``None``."""
        row = self._connection.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict[str, object]]:
        """Return all users ordered by last login desc."""
        rows = self._connection.execute(
            "SELECT * FROM users ORDER BY last_login DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- document ownership ---------------------------------------------------

    def set_document_owner(
        self,
        doc_id: str,
        user_id: str,
        display_name: str,
    ) -> None:
        """Record (or transfer) ownership of *doc_id* to *user_id*."""
        with self._lock:
            conn = self._connection
            conn.execute(
                """
                INSERT INTO document_owners (doc_id, owner_id, owner_name, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    owner_name=excluded.owner_name,
                    updated_at=excluded.updated_at
                """,
                (doc_id, user_id, display_name, time.time()),
            )
            conn.commit()

    def get_document_owner(self, doc_id: str) -> dict[str, object] | None:
        """Return the owner row for *doc_id*, or ``None``."""
        row = self._connection.execute(
            "SELECT * FROM document_owners WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_owners_for_documents(self, doc_ids: list[str]) -> dict[str, dict[str, object]]:
        """Return ``{doc_id: owner_row}`` for every requested doc_id."""
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        rows = self._connection.execute(
            f"SELECT * FROM document_owners WHERE doc_id IN ({placeholders})",  # noqa: S608
            doc_ids,
        ).fetchall()
        return {str(r["doc_id"]): dict(r) for r in rows}

    def delete_document_owner(self, doc_id: str) -> None:
        """Remove the ownership record for *doc_id*."""
        with self._lock:
            conn = self._connection
            conn.execute("DELETE FROM document_owners WHERE doc_id = ?", (doc_id,))
            conn.commit()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    username     TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL DEFAULT '',
    avatar       TEXT,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    last_login   REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS document_owners (
    doc_id     TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    owner_name TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);
"""
