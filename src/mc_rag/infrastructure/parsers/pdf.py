"""PDF document parser built on top of :mod:`pypdf`.

The public entry point is :func:`parse_pdf`, which returns a normalized
:class:`~mc_rag.infrastructure.parsers.base.ParsedDocument`. Text is extracted
page by page and joined with blank lines; metadata exposed by the PDF
(title, author, page count, ...) is surfaced for later citation and filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from mc_rag.domain.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
)
from mc_rag.infrastructure.parsers.base import ParsedDocument, resolve_document_path

logger = logging.getLogger(__name__)

_FORMAT = "pdf"


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """Extract text and metadata from a PDF file.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A :class:`ParsedDocument` containing the concatenated text of every
        page together with basic metadata (title, author, page count).

    Raises:
        DocumentNotFoundError: The path does not exist or is not a file.
        EmptyDocumentError: The file is zero bytes or yields no text at all.
        EncryptedDocumentError: The PDF is encrypted and cannot be opened.
        CorruptDocumentError: The PDF is malformed or otherwise unreadable.
    """
    path = resolve_document_path(file_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        logger.warning("Failed to open PDF %s: %s", path, exc)
        raise CorruptDocumentError(f"Unable to open PDF: {path}") from exc

    if reader.is_encrypted and not _try_decrypt(reader):
        raise EncryptedDocumentError(f"PDF is encrypted and could not be decrypted: {path}")

    pages_text = _extract_pages_text(reader, path)
    text = "\n\n".join(pages_text).strip()
    if not text:
        raise EmptyDocumentError(
            f"PDF contains no extractable text (possibly a scanned image): {path}"
        )

    metadata = _extract_metadata(reader)
    metadata["page_count"] = len(pages_text)

    return ParsedDocument(
        text=text,
        source=str(path),
        format=_FORMAT,
        title=_coerce_optional_str(metadata.get("title")),
        metadata=metadata,
    )


def _try_decrypt(reader: PdfReader) -> bool:
    """Attempt to open an encrypted PDF using the empty user password."""
    try:
        return bool(reader.decrypt(""))
    except Exception as exc:  # noqa: BLE001 - pypdf raises varied errors
        logger.debug("Empty-password decryption failed: %s", exc)
        return False


def _extract_pages_text(reader: PdfReader, path: Path) -> list[str]:
    """Extract text from every page, tolerating per-page failures."""
    pages_text: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            logger.warning("Failed to extract text from %s page %d: %s", path, index + 1, exc)
            pages_text.append("")
    return pages_text


def _extract_metadata(reader: PdfReader) -> dict[str, Any]:
    """Collect the descriptive fields exposed by the PDF document info."""
    info = reader.metadata
    if info is None:
        return {}

    fields = {
        "title": info.title,
        "author": info.author,
        "subject": info.subject,
        "creator": info.creator,
        "producer": info.producer,
    }
    return {key: str(value) for key, value in fields.items() if value}


def _coerce_optional_str(value: Any) -> str | None:
    """Return a stripped string for *value*, or ``None`` when it is blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
