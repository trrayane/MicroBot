from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mc_rag.api.dependencies import (
    Access,
    get_query_log_repository,
    require_admin,
)
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository
from mc_rag.main import app


def test_repetitive_questions_requires_admin_and_applies_threshold(tmp_path: Path) -> None:
    repository = QueryLogRepository(tmp_path / "queries.sqlite3")
    for _ in range(10):
        query_id = repository.add_query("conversation-a", "Where is the club?")
        if query_id % 3 == 0:
            repository.set_feedback(
                "down", query_log_id=query_id, conversation_id="conversation-a"
            )
    repository.add_query("conversation-b", "How do I join?")

    client = TestClient(app)
    assert client.get("/api/v1/admin/repetitive-questions").status_code == 401

    app.dependency_overrides[get_query_log_repository] = lambda: repository
    app.dependency_overrides[require_admin] = lambda: Access(is_member=True, is_admin=True)
    try:
        response = client.get("/api/v1/admin/repetitive-questions")
        assert response.status_code == 200
        assert response.json() == {
            "questions": [
                {
                    "normalized_query": "where is the club",
                    "total_count": 10,
                    "unuseful_count": 3,
                }
            ]
        }

        filtered = client.get(
            "/api/v1/admin/repetitive-questions", params={"min_frequency": 11}
        )
        assert filtered.status_code == 200
        assert filtered.json() == {"questions": []}
    finally:
        app.dependency_overrides.clear()