"""Unit tests for :mod:`mc_rag.application.ingestion_status`."""

from __future__ import annotations

from mc_rag.application.ingestion_status import IngestionStatusTracker


def test_start_finish_and_get_roundtrip() -> None:
    tracker = IngestionStatusTracker()

    job = tracker.start("doc-1", title="Guide", visibility="public")
    assert job.status == "processing"
    assert tracker.get("doc-1") is job

    tracker.finish("doc-1", status="indexed", chunks=7)
    stored = tracker.get("doc-1")
    assert stored is not None
    assert stored.status == "indexed"
    assert stored.chunks == 7
    assert stored.finished_at is not None


def test_progress_updates_stage_and_chunk_counts() -> None:
    tracker = IngestionStatusTracker()
    tracker.start("doc-2", title="Guide", visibility="public")

    tracker.progress("doc-2", stage="parsing")
    tracker.progress("doc-2", stage="embedding", chunks_done=16, chunks_total=32)
    tracker.progress("doc-2", stage="embedding", chunks_done=32, chunks_total=32)
    tracker.progress("doc-2", stage="indexing", chunks_done=32, chunks_total=32)

    job = tracker.get("doc-2")
    assert job is not None
    assert job.status == "processing"
    assert job.stage == "indexing"
    assert job.chunks_done == 32
    assert job.chunks_total == 32


def test_progress_never_regresses_and_ignores_unknown_jobs() -> None:
    tracker = IngestionStatusTracker()
    tracker.start("doc-3", title="Guide", visibility="public")

    tracker.progress("doc-3", stage="embedding", chunks_done=10, chunks_total=20)
    tracker.progress("doc-3", stage="embedding", chunks_done=4, chunks_total=20)
    tracker.progress("unknown-doc", stage="embedding", chunks_done=5, chunks_total=5)

    job = tracker.get("doc-3")
    assert job is not None
    assert job.chunks_done == 10
    assert tracker.get("unknown-doc") is None


def test_progress_is_ignored_once_the_job_is_terminal() -> None:
    tracker = IngestionStatusTracker()
    tracker.start("doc-4", title="Guide", visibility="public")
    tracker.finish("doc-4", status="indexed", chunks=3)

    tracker.progress("doc-4", stage="embedding", chunks_done=99, chunks_total=99)

    job = tracker.get("doc-4")
    assert job is not None
    assert job.stage == "queued"
    assert job.chunks_done == 0
    assert job.chunks_total == 0


def test_finished_jobs_are_evicted_before_running_ones() -> None:
    tracker = IngestionStatusTracker(max_entries=2)
    tracker.start("old-done", title="A", visibility="public")
    tracker.finish("old-done", status="indexed")
    tracker.start("running", title="B", visibility="public")
    tracker.start("newest", title="C", visibility="public")

    assert tracker.get("old-done") is None
    assert tracker.get("running") is not None
    assert tracker.get("newest") is not None
