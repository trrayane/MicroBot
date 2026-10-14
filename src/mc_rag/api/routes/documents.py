"""Document management endpoints (admin only): list, add, delete.

Adding a document accepts three modes, all sent as ``multipart/form-data`` so a
file upload and the text-based modes share one endpoint:

* ``mode=file`` -> ``file`` (PDF, DOCX, Markdown, TXT, XLSX or CSV), stored under
  ``UPLOAD_DIR`` then ingested;
* ``mode=url``  -> ``url`` fetched and cleaned server side (Google Sheets links
    are exported as a spreadsheet, everything else goes through the SSRF guard);
* ``mode=text`` -> ``text`` pasted directly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from mc_rag.api.dependencies import (
    get_ingestion_service,
    get_lexical_index,
    get_store,
    require_admin,
)
from mc_rag.api.schemas import (
    AddDocumentResponse,
    DeleteDocumentResponse,
    DocumentOut,
    DocumentsResponse,
)
from mc_rag.application.ingestion_service import FileIngestionResult, IngestionService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import DomainError
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index
from mc_rag.infrastructure.parsers import (
    ParsedDocument,
    is_google_sheets_url,
    parse_google_sheets,
    parse_web,
)
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


def _result_response(
    result: FileIngestionResult, *, title: str, visibility: str
) -> AddDocumentResponse:
    if not result.succeeded:
        raise HTTPException(400, result.error or "Ingestion failed.")
    return AddDocumentResponse(
        doc_id=result.doc_id or "",
        title=title,
        status=str(result.status),
        chunks=result.chunks,
        visibility=visibility,
    )


@router.get("", response_model=DocumentsResponse)
def list_documents(store: Annotated[ChromaStore, Depends(get_store)]) -> DocumentsResponse:
    """List every indexed document with its chunk count."""
    rows = store.list_documents()
    return DocumentsResponse(documents=[DocumentOut(**row) for row in rows])


@router.post("", response_model=AddDocumentResponse)
async def add_document(
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    mode: Annotated[str, Form()],
    visibility: Annotated[str, Form()] = VISIBILITY_PUBLIC,
    title: Annotated[str | None, Form()] = None,
    url: Annotated[str | None, Form()] = None,
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> AddDocumentResponse:
    """Ingest a file, a web article or pasted text."""
    _check_visibility(visibility)
    settings = get_settings()
    mode = mode.strip().lower()

    if mode == "file":
        return await _add_file(file, title, visibility, service, settings.upload_dir)
    if mode == "url":
        return _add_url(url, visibility, service)
    if mode == "text":
        return _add_text(text, title, visibility, service)
    raise HTTPException(400, f"Unknown mode {mode!r}; expected 'file', 'url' or 'text'.")


async def _add_file(
    file: UploadFile | None,
    title: str | None,
    visibility: str,
    service: IngestionService,
    upload_dir: str,
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

    if suffix in _TEXT_SUFFIXES:
        document = ParsedDocument(
            text=data.decode("utf-8", "replace"),
            source=str(target),
            format="text" if suffix == ".txt" else "markdown",
            title=doc_title,
            metadata={"visibility": visibility},
        )
        doc_id = service.stable_doc_id(f"upload:{visibility}/{target.name}")
        result = service.ingest_parsed_document(document, doc_id=doc_id, visibility=visibility)
    else:
        result = service.ingest_file(target, root=upload_root, visibility=visibility)
    return _result_response(result, title=doc_title, visibility=visibility)


def _add_url(url: str | None, visibility: str, service: IngestionService) -> AddDocumentResponse:
    clean = (url or "").strip()
    if not clean:
        raise HTTPException(400, "No URL provided.")
    try:
        # ``allow_local_files=False``: a submitted URL must never read a local file.
        if is_google_sheets_url(clean):
            document = parse_google_sheets(clean)
        else:
            document = parse_web(clean, allow_local_files=False)
    except (DomainError, OSError) as exc:
        raise HTTPException(400, f"Could not read this page: {exc}") from exc
    doc_id = service.stable_doc_id(f"url:{clean}")
    result = service.ingest_parsed_document(document, doc_id=doc_id, visibility=visibility)
    title = document.title or clean
    return _result_response(result, title=title, visibility=visibility)


def _add_text(
    text: str | None, title: str | None, visibility: str, service: IngestionService
) -> AddDocumentResponse:
    content = (text or "").strip()
    if not content:
        raise HTTPException(400, "Text is empty.")
    doc_title = (title or "").strip() or "Texte collé"
    document = ParsedDocument(
        text=content,
        source=f"text:{doc_title}",
        format="text",
        title=doc_title,
        metadata={"visibility": visibility},
    )
    doc_id = service.stable_doc_id(f"text:{doc_title}")
    result = service.ingest_parsed_document(document, doc_id=doc_id, visibility=visibility)
    return _result_response(result, title=doc_title, visibility=visibility)


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    store: Annotated[ChromaStore, Depends(get_store)],
    lexical: Annotated[BM25Index, Depends(get_lexical_index)],
) -> DeleteDocumentResponse:
    """Remove every chunk of *doc_id* from both indexes."""
    removed = store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, "Document not found.")
    try:
        lexical.delete_document(doc_id)
    except (DomainError, OSError) as exc:
        # The vector store is the source of truth; rebuild BM25 if it drifts.
        logger.warning("Lexical index not cleaned for %s: %s", doc_id, exc)
    return DeleteDocumentResponse(doc_id=doc_id, chunks_deleted=removed)
