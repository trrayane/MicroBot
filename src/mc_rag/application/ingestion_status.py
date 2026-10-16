"""Thread-safe in-memory tracker for background ingestion jobs.

Lets the API return ``202 Accepted`` immediately and lets the UI poll
``GET /documents/{doc_id}/status`` until the job reaches a terminal state.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class IngestionJob:
    """State of one background ingestion job."""

    doc_id: str
    title: str
    visibility: str
    status: str = "processing"  # processing | indexed | updated | unchanged | failed
    chunks: int = 0
    error: str | None = None
    # Progress within the job so a polling client can show something useful
    # instead of a spinner that appears frozen for minutes.
    stage: str = "queued"  # queued | parsing | chunking | embedding | indexing
    chunks_done: int = 0
    chunks_total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def elapsed_ms(self) -> int:
        """Milliseconds since the job started."""
        end = self.finished_at if self.finished_at is not None else time.time()
        return int((end - self.started_at) * 1000)


class IngestionStatusTracker:
    """Process-wide, thread-safe store of ingestion job statuses."""

    def __init__(self, *, max_entries: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._max = max_entries

    def start(self, doc_id: str, *, title: str, visibility: str) -> IngestionJob:
        """Register a new job (or reset an existing one for the same doc_id)."""
        job = IngestionJob(doc_id=doc_id, title=title, visibility=visibility)
        with self._lock:
            self._jobs[doc_id] = job
            self._evict_locked()
        return job

    def progress(
        self,
        doc_id: str,
        *,
        stage: str,
        chunks_done: int = 0,
        chunks_total: int = 0,
    ) -> None:
        """Update the stage/progress of a running job (ignored once terminal)."""
        with self._lock:
            job = self._jobs.get(doc_id)
            if job is None or job.finished_at is not None:
                return
            job.stage = stage
            if chunks_done:
                job.chunks_done = max(job.chunks_done, chunks_done)
            if chunks_total:
                job.chunks_total = max(job.chunks_total, chunks_total)

    def finish(
        self,
        doc_id: str,
        *,
        status: str,
        chunks: int = 0,
        error: str | None = None,
    ) -> None:
        """Mark a job as terminal."""
        with self._lock:
            job = self._jobs.get(doc_id)
            if job is None:
                return
            job.status = status
            job.chunks = chunks
            job.error = error
            job.finished_at = time.time()

    def get(self, doc_id: str) -> IngestionJob | None:
        """Return the job for *doc_id*, or ``None`` when unknown."""
        with self._lock:
            return self._jobs.get(doc_id)

    def _evict_locked(self) -> None:
        """Keep at most ``max_entries`` jobs (drop finished ones first)."""
        if len(self._jobs) <= self._max:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.finished_at is not None),
            key=lambda job: job.finished_at or 0.0,
        )
        for job in finished:
            if len(self._jobs) <= self._max:
                break
            self._jobs.pop(job.doc_id, None)


_tracker: IngestionStatusTracker | None = None
_tracker_lock = threading.Lock()


def get_status_tracker() -> IngestionStatusTracker:
    """Return the process-wide tracker (lazily created)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = IngestionStatusTracker()
    return _tracker
