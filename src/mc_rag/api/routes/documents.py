"""Document management endpoints (admin only): list, add, delete, status.

Adding a document accepts three modes, all sent as ``multipart/form-data``:

* ``mode=file`` -> ``file`` (PDF, DOCX, Markdown, TXT, XLSX or CSV);
* ``mode=url``  -> ``url`` fetched and cleaned server side;
* ``mode=text`` -> ``text`` pasted directly.

All three modes are **asynchronous**: the request returns ``202 Accepted``
immediately and the heavy work (parse → chunk → embed → index) runs in a
background task.  The client polls ``GET /documents/{doc_id}/status`` until
the job reaches a terminal state (``indexed``, ``updated``, ``unchanged``
or ``failed``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from mc_rag.api.dependencies import (
    Access,
    get_access,
    get_ingestion_service,
    get_lexical_index,
    get_store,
    get_user_store,
    require_admin,
)
from mc_rag.api.schemas import (
    AddDocumentResponse,
    DeleteDocumentResponse,
    DocumentOut,
    DocumentsResponse,
    DocumentStatusResponse,
)
from mc_rag.application.ingestion_service import IngestionService
from mc_rag.application.ingestion_status import IngestionStatusTracker, get_status_tracker
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import DomainError
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index
from mc_rag.infrastructure.parsers import (
    ParsedDocument,
    is_google_sheets_url,
    parse_google_sheets,
    parse_web,
)
from mc_rag.infrastructure.persistence.user_store import UserStore
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_PUBLIC,
    ChromaStore,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_admin)])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_FILE_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".xlsx", ".csv"}
_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    """Reduce a client-supplied filename to a safe basename."""
    cleaned = _UNSAFE_NAME.sub("_", Path(name).name).strip("._")
    return cleaned or "upload"


def _check_visibility(visibility: str) -> str:
    if visibility not in ALL_VISIBILITIES:
        raise HTTPException(400, f"Unknown visibility {visibility!r}.")
    return visibility


@router.get("", response_model=DocumentsResponse)
def list_documents(
    store: Annotated[ChromaStore, Depends(get_store)],
    user_store: Annotated[UserStore, Depends(get_user_store)],
) -> DocumentsResponse:
    """List every indexed document with its chunk count and uploader."""
    rows = store.list_documents()
    doc_ids = [row["doc_id"] for row in rows]
    owners = user_store.get_owners_for_documents(doc_ids)
    documents = []
    for row in rows:
        owner = owners.get(row["doc_id"])
        documents.append(
            DocumentOut(
                **row,
                uploaded_by=str(owner["owner_name"]) if owner else None,
            )
        )
    return DocumentsResponse(documents=documents)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
def document_status(
    doc_id: str,
    tracker: Annotated[IngestionStatusTracker, Depends(get_status_tracker)],
) -> DocumentStatusResponse:
    """Return the ingestion status for *doc_id* (poll this after an upload)."""
    job = tracker.get(doc_id)
    if job is None:
        raise HTTPException(404, "Unknown document or job expired.")
    return DocumentStatusResponse(
        doc_id=job.doc_id,
        title=job.title,
        status=job.status,
        chunks=job.chunks,
        error=job.error,
        elapsed_ms=job.elapsed_ms,
    )


@router.post("", response_model=AddDocumentResponse, status_code=202)
async def add_document(
    background: BackgroundTasks,
    access: Annotated[Access, Depends(get_access)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    tracker: Annotated[IngestionStatusTracker, Depends(get_status_tracker)],
    user_store: Annotated[UserStore, Depends(get_user_store)],
    mode: Annotated[str, Form()],
    visibility: Annotated[str, Form()] = VISIBILITY_PUBLIC,
    title: Annotated[str | None, Form()] = None,
    url: Annotated[str | None, Form()] = None,
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> AddDocumentResponse:
    """Queue a file, a web article or pasted text for background ingestion."""
    _check_visibility(visibility)
    settings = get_settings()
    mode = mode.strip().lower()

    # Determine the uploader for attribution.
    uploader_id = access.user_id or "admin-key"
    uploader_name = access.display_name or "Admin"

    if mode == "file":
        return await _queue_file(
            background, file, title, visibility, service, tracker,
            settings.upload_dir, user_store, uploader_id, uploader_name,
        )
    if mode == "url":
        return _queue_url(
            background, url, visibility, service, tracker,
            user_store, uploader_id, uploader_name,
        )
    if mode == "text":
        return _queue_text(
            background, text, title, visibility, service, tracker,
            user_store, uploader_id, uploader_name,
        )
    raise HTTPException(400, f"Unknown mode {mode!r}; expected 'file', 'url' or 'text'.")


# --------------------------------------------------------------------------- #
# File mode
# --------------------------------------------------------------------------- #


async def _queue_file(
    background: BackgroundTasks,
    file: UploadFile | None,
    title: str | None,
    visibility: str,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    upload_dir: str,
    user_store: UserStore,
    uploader_id: str,
    uploader_name: str,
) -> AddDocumentResponse:
    if file is None or not file.filename:
        raise HTTPException(400, "No file provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _FILE_SUFFIXES:
        raise HTTPException(400, "Unsupported file type (PDF, DOCX, MD, TXT, XLSX or CSV).")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (10 MB maximum).")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    upload_root = Path(upload_dir)
    target_dir = upload_root / visibility
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _safe_filename(file.filename)
    target.write_bytes(data)
    doc_title = (title or "").strip() or target.stem
    doc_id = service.stable_doc_id(f"upload:{visibility}/{target.name}")

    # Record ownership for the admin table.
    user_store.set_document_owner(doc_id, uploader_id, uploader_name)

    tracker.start(doc_id, title=doc_title, visibility=visibility)
    background.add_task(
        _run_ingestion,
        service=service,
        tracker=tracker,
        doc_id=doc_id,
        title=doc_title,
        visibility=visibility,
        path=target,
        root=upload_root,
        is_text=suffix in _TEXT_SUFFIXES,
        raw_text=data.decode("utf-8", "replace") if suffix in _TEXT_SUFFIXES else None,
        fmt="text" if suffix == ".txt" else ("markdown" if suffix in _TEXT_SUFFIXES else None),
    )
    return AddDocumentResponse(
        doc_id=doc_id, title=doc_title, status="processing", chunks=0, visibility=visibility
    )


# --------------------------------------------------------------------------- #
# URL mode
# --------------------------------------------------------------------------- #


def _queue_url(
    background: BackgroundTasks,
    url: str | None,
    visibility: str,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    user_store: UserStore,
    uploader_id: str,
    uploader_name: str,
) -> AddDocumentResponse:
    clean = (url or "").strip()
    if not clean:
        raise HTTPException(400, "No URL provided.")
    doc_id = service.stable_doc_id(f"url:{clean}")
    doc_title = clean
    user_store.set_document_owner(doc_id, uploader_id, uploader_name)
    tracker.start(doc_id, title=doc_title, visibility=visibility)
    background.add_task(
        _run_url_ingestion,
        service=service,
        tracker=tracker,
        doc_id=doc_id,
        visibility=visibility,
        url=clean,
    )
    return AddDocumentResponse(
        doc_id=doc_id, title=doc_title, status="processing", chunks=0, visibility=visibility
    )


def _run_url_ingestion(
    *,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    doc_id: str,
    visibility: str,
    url: str,
) -> None:
    """Background task: fetch, parse, chunk, embed and index a URL."""
    try:
        if is_google_sheets_url(url):
            document = parse_google_sheets(url)
        else:
            document = parse_web(url, allow_local_files=False)
        result = service.ingest_parsed_document(document, doc_id=doc_id, visibility=visibility)
        tracker.finish(
            doc_id, status=str(result.status), chunks=result.chunks, error=result.error
        )
    except Exception as exc:
        logger.exception("Background URL ingestion failed for %s", url)
        tracker.finish(doc_id, status="failed", error=str(exc))


# --------------------------------------------------------------------------- #
# Text mode
# --------------------------------------------------------------------------- #


def _queue_text(
    background: BackgroundTasks,
    text: str | None,
    title: str | None,
    visibility: str,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    user_store: UserStore,
    uploader_id: str,
    uploader_name: str,
) -> AddDocumentResponse:
    content = (text or "").strip()
    if not content:
        raise HTTPException(400, "Text is empty.")
    doc_title = (title or "").strip() or "Texte collé"
    doc_id = service.stable_doc_id(f"text:{doc_title}")
    user_store.set_document_owner(doc_id, uploader_id, uploader_name)
    tracker.start(doc_id, title=doc_title, visibility=visibility)
    background.add_task(
        _run_text_ingestion,
        service=service,
        tracker=tracker,
        doc_id=doc_id,
        visibility=visibility,
        content=content,
        doc_title=doc_title,
    )
    return AddDocumentResponse(
        doc_id=doc_id, title=doc_title, status="processing", chunks=0, visibility=visibility
    )


def _run_text_ingestion(
    *,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    doc_id: str,
    visibility: str,
    content: str,
    doc_title: str,
) -> None:
    """Background task: parse, chunk, embed and index pasted text."""
    try:
        document = ParsedDocument(
            text=content,
            source=f"text:{doc_title}",
            format="text",
            title=doc_title,
            metadata={"visibility": visibility},
        )
        result = service.ingest_parsed_document(document, doc_id=doc_id, visibility=visibility)
        tracker.finish(
            doc_id, status=str(result.status), chunks=result.chunks, error=result.error
        )
    except Exception as exc:
        logger.exception("Background text ingestion failed for %s", doc_title)
        tracker.finish(doc_id, status="failed", error=str(exc))


# --------------------------------------------------------------------------- #
# Shared background worker
# --------------------------------------------------------------------------- #


def _run_ingestion(
    *,
    service: IngestionService,
    tracker: IngestionStatusTracker,
    doc_id: str,
    title: str,
    visibility: str,
    path: Path,
    root: Path,
    is_text: bool,
    raw_text: str | None,
    fmt: str | None,
) -> None:
    """Background task: parse, chunk, embed and index a saved file."""
    try:
        if is_text and raw_text is not None:
            document = ParsedDocument(
                text=raw_text,
                source=str(path),
                format=fmt or "text",
                title=title,
                metadata={"visibility": visibility},
            )
            result = service.ingest_parsed_document(
                document, doc_id=doc_id, visibility=visibility
            )
        else:
            result = service.ingest_file(path, root=root, visibility=visibility)
        tracker.finish(
            doc_id, status=str(result.status), chunks=result.chunks, error=result.error
        )
    except Exception as exc:
        logger.exception("Background ingestion failed for %s", path)
        tracker.finish(doc_id, status="failed", error=str(exc))


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    store: Annotated[ChromaStore, Depends(get_store)],
    lexical: Annotated[BM25Index, Depends(get_lexical_index)],
    user_store: Annotated[UserStore, Depends(get_user_store)],
) -> DeleteDocumentResponse:
    """Remove every chunk of *doc_id* from both indexes."""
    removed = store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, "Document not found.")
    try:
        lexical.delete_document(doc_id)
    except (DomainError, OSError) as exc:
        logger.warning("Lexical index not cleaned for %s: %s", doc_id, exc)
    user_store.delete_document_owner(doc_id)
    return DeleteDocumentResponse(doc_id=doc_id, chunks_deleted=removed)
