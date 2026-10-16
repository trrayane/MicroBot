"""SQLite persistence for normalized chat queries and their feedback."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class RepetitiveQuestion:
    normalized_query: str
    total_count: int
    unuseful_count: int


def normalize_query(query: str) -> str:
    """Normalize whitespace, punctuation, and case for grouping similar questions."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", normalized).split())


class QueryLogRepository:
    """Stores successful chat queries and feedback on their answers."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add_query(self, conversation_id: str, query: str) -> int:
        normalized = normalize_query(query)
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO query_logs (conversation_id, normalized_query) VALUES (?, ?)",
                (conversation_id, normalized),
            )
            return int(cursor.lastrowid)

    def set_feedback(
        self,
        vote: str,
        *,
        query_log_id: int | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """Attach a vote to its query, falling back to the conversation's latest query."""
        feedback = "not_useful" if vote in {"down", "not_useful"} else "useful"
        with closing(self._connect()) as conn, conn:
            if query_log_id is not None and conversation_id is not None:
                conn.execute(
                    "UPDATE query_logs SET feedback = ? WHERE id = ? AND conversation_id = ?",
                    (feedback, query_log_id, conversation_id),
                )
            elif query_log_id is None and conversation_id is not None:
                conn.execute(
                    "UPDATE query_logs SET feedback = ? WHERE id = "
                    "(SELECT id FROM query_logs WHERE conversation_id = ? "
                    "ORDER BY id DESC LIMIT 1)",
                    (feedback, conversation_id),
                )

    def list_repetitive(self, *, min_frequency: int = 10) -> list[RepetitiveQuestion]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT normalized_query, COUNT(*) AS total_count, "
                "SUM(CASE WHEN feedback = 'not_useful' THEN 1 ELSE 0 END) "
                "AS unuseful_count FROM query_logs "
                "GROUP BY normalized_query HAVING COUNT(*) >= ? "
                "ORDER BY total_count DESC, unuseful_count DESC, normalized_query ASC",
                (min_frequency,),
            ).fetchall()
        return [RepetitiveQuestion(row[0], row[1], row[2]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    feedback TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_query_logs_normalized_query
                    ON query_logs (normalized_query);
                CREATE INDEX IF NOT EXISTS idx_query_logs_conversation
                    ON query_logs (conversation_id, id);
                """
            )