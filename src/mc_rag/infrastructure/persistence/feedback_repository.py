"""SQLite persistence for user feedback (stdlib ``sqlite3``)."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class FeedbackRecord:
    """A single stored feedback entry."""

    feedback_id: str
    conversation_id: str | None
    vote: str
    created_at: float


class FeedbackRepository:
    """Stores chat feedback. One short-lived connection per call."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add(self, *, conversation_id: str | None, vote: str) -> FeedbackRecord:
        """Persist one vote and return the stored record."""
        feedback_id = f"f_{uuid.uuid4().hex[:12]}"
        created_at = time.time()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO feedback (id, conversation_id, vote, created_at) "
                "VALUES (?, ?, ?, ?)",
                (feedback_id, conversation_id, vote, created_at),
            )
        return FeedbackRecord(feedback_id, conversation_id, vote, created_at)

    def list(self, *, limit: int = 200) -> list[FeedbackRecord]:
        """Return the most recent feedback entries, newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, vote, created_at FROM feedback "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [FeedbackRecord(row[0], row[1], row[2], row[3]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    def _init_schema(self) -> None:
        with closing(sqlite3.connect(self._db_path, timeout=10)) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    vote TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
