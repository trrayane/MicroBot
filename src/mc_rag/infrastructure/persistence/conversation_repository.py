"""SQLite persistence for conversations and their messages (stdlib ``sqlite3``)."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ConversationRecord:
    """A stored conversation. ``is_member`` is True once a member has used it."""

    conversation_id: str
    is_member: bool


class ConversationRepository:
    """Stores conversations and messages. One short-lived connection per call."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create(self, *, is_member: bool) -> str:
        conversation_id = f"c_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, is_member, created_at) VALUES (?, ?, ?)",
                (conversation_id, int(is_member), time.time()),
            )
        return conversation_id

    def get(self, conversation_id: str) -> ConversationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, is_member FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return ConversationRecord(row[0], bool(row[1])) if row else None

    def mark_member(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET is_member = 1 WHERE id = ?", (conversation_id,))

    def add_messages(self, conversation_id: str, messages: list[tuple[str, str]]) -> None:
        """Append ``(role, content)`` pairs in order."""
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                [(conversation_id, role, content, now) for role, content in messages],
            )

    def recent_messages(self, conversation_id: str, limit: int) -> list[dict[str, str]]:
        """Return the last *limit* messages, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def _connect(self) -> closing[sqlite3.Connection] | sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    def _init_schema(self) -> None:
        with closing(sqlite3.connect(self._db_path, timeout=10)) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    is_member INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, id);
                """
            )