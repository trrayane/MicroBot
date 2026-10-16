"""API tests for the asynchronous document upload pipeline (admin only)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient

from mc_rag.api.dependencies import (
    Access,
    get_ingestion_service,
    get_query_log_repository,
    get_store,
    get_user_store,
    require_admin,
)
from mc_rag.application.ingestion_service import IngestionService
from mc_rag.config import get_settings
from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index
from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository
from mc_rag.infrastructure.persistence.user_store import UserStore
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore
from mc_rag.main import app
from tests.integration.test_ingestion_service import HashingModel

_DIMENSION = 8
_BODY = b"# Note\n\n" + b"Useful sentence about the club. " * 200


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """TestClient wired to a fast, in-memory ingestion stack."""
    store = ChromaStore(client=chromadb.EphemeralClient(), collection_name=f"api_{tmp_path.name}")
    lexical = BM25Index(path=tmp_path / "bm25.sqlite3")
    service = IngestionService(
        store=store,
        embedder=LocalEmbedder(model=HashingModel(), dimension=_DIMENSION),
        lexical=lexical,
        raw_data_dir=tmp_path / "raw",
    )
    user_store = UserStore(tmp_path / "users.db")
    repository = QueryLogRepository(tmp_path / "queries.sqlite3")

    settings = get_settings()
    original_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path / "uploads")

    app.dependency_overrides[get_ingestion_service] = lambda: service
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_user_store] = lambda: user_store
    app.dependency_overrides[get_query_log_repository] = lambda: repository
    app.dependency_overrides[require_admin] = lambda: Access(is_member=True, is_admin=True)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        settings.upload_dir = original_upload_dir


def _upload(client: TestClient, name: str, content: bytes, **form: str):
    return client.post(
        "/api/v1/documents",
        data={"mode": "file", "visibility": "public", **form},
        files={"file": (name, content, "application/octet-stream")},
    )


def test_upload_returns_202_immediately_and_indexes_in_the_background(client: TestClient) -> None:
    response = _upload(client, "note.md", _BODY)

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "processing"
    assert payload["chunks"] == 0
    doc_id = payload["doc_id"]

    # The background task has run by the time TestClient returns: the status
    # endpoint now reports a terminal state with the real chunk count and the
    # last stage it reached.
    status = client.get(f"/api/v1/documents/{doc_id}/status").json()
    assert status["status"] == "indexed"
    assert status["chunks"] > 0
    assert status["stage"] == "indexing"
    assert status["chunks_done"] == status["chunks_total"] == status["chunks"]
    assert status["error"] is None

    listed = client.get("/api/v1/documents").json()["documents"]
    assert [row["doc_id"] for row in listed] == [doc_id]
    # The uploader comes from ``get_access`` (no credentials in this test ->
    # the "Admin" placeholder), and ownership survives in the user store.
    assert listed[0]["uploaded_by"] == "Admin"


def test_upload_records_ownership_and_reingest_reports_unchanged(client: TestClient) -> None:
    first = _upload(client, "again.md", _BODY).json()
    assert first["status"] == "processing"

    second = _upload(client, "again.md", _BODY).json()
    assert second["doc_id"] == first["doc_id"]

    status = client.get(f"/api/v1/documents/{second['doc_id']}/status").json()
    assert status["status"] == "unchanged"


def test_empty_and_unsupported_uploads_are_rejected(client: TestClient) -> None:
    assert _upload(client, "empty.md", b"").status_code == 400
    assert _upload(client, "malware.exe", b"MZ").status_code == 400

    unknown = client.get("/api/v1/documents/does-not-exist/status")
    assert unknown.status_code == 404


def test_oversized_upload_is_rejected_and_leaves_no_file(client: TestClient) -> None:
    too_big = b"x" * (10 * 1024 * 1024 + 1)
    response = _upload(client, "big.md", too_big)

    assert response.status_code == 413
    upload_dir = Path(get_settings().upload_dir) / "public"
    assert not (upload_dir / "big.md").exists()


def test_repetitive_questions_dashboard_data(client: TestClient) -> None:
    """End-to-end check of the aggregation the admin dashboard renders."""
    repository = app.dependency_overrides[get_query_log_repository]()
    for _ in range(10):
        query_id = repository.add_query("conversation-a", "How do I join the club?")
        repository.set_feedback("down", query_log_id=query_id, conversation_id="conversation-a")
    repository.add_query("conversation-b", "How do I join the club?")
    repository.add_query("conversation-c", "What are the fees?")

    response = client.get("/api/v1/admin/repetitive-questions", params={"min_frequency": 10})

    assert response.status_code == 200
    assert response.json() == {
        "questions": [
            {"normalized_query": "how do i join the club", "total_count": 11, "unuseful_count": 10}
        ]
    }
