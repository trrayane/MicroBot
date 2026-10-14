"""CSV parser built on the standard library :mod:`csv` module.

The public entry points are :func:`parse_csv` (a file on disk) and
:func:`parse_csv_text` (an already decoded payload, used for Google Sheets).
Semicolons and commas are both understood, a UTF-8 BOM is stripped, and legacy
``cp1252`` exports are decoded when the file is not valid UTF-8.

Each data row becomes one rendered line with its headers, e.g.
``Fichier : bureau.csv | Nom : Amine | Rôle : Trésorier``.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import CorruptDocumentError, EmptyDocumentError, ParsingError
from mc_rag.infrastructure.parsers.base import ParsedDocument, TableRow, resolve_document_path
from mc_rag.infrastructure.parsers.table import build_rows

logger = logging.getLogger(__name__)

_FORMAT = "csv"
_SNIFFER_DELIMITERS = ";,\t|"
_SNIFF_CHARS = 8192
_FALLBACK_DELIMITERS = (";", ",", "\t", "|")


def parse_csv(
    file_path: str | Path,
    *,
    max_rows: int | None = None,
    max_cell_chars: int | None = None,
) -> ParsedDocument:
    """Extract a CSV file as rendered rows.

    Raises:
        DocumentNotFoundError: The path does not exist or is not a file.
        EmptyDocumentError: The file is zero bytes or holds no data row.
        CorruptDocumentError: The file cannot be decoded at all.
    """
    path = resolve_document_path(file_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorruptDocumentError(f"Unable to read CSV file: {path}") from exc
    text = _decode(raw, path=str(path))
    return parse_csv_text(
        text, source=str(path), title=path.stem, max_rows=max_rows, max_cell_chars=max_cell_chars
    )


def parse_csv_text(
    text: str,
    *,
    source: str,
    title: str | None = None,
    max_rows: int | None = None,
    max_cell_chars: int | None = None,
    file_name: str | None = None,
) -> ParsedDocument:
    """Parse an in-memory CSV payload into rendered rows.

    Args:
        text: The decoded CSV content.
        source: Provenance recorded on the document (path or URL).
        title: Document title; defaults to ``None``.
        max_rows: Row budget (defaults to ``SHEET_MAX_ROWS``).
        max_cell_chars: Per-cell character cap.
        file_name: Name rendered inside every row; defaults to the ``source``
            base name.
    """
    if not text.strip():
        raise EmptyDocumentError(f"CSV file is empty: {source}")

    settings = get_settings()
    row_limit = max_rows if max_rows is not None else settings.sheet_max_rows
    cell_cap = max_cell_chars if max_cell_chars is not None else settings.sheet_max_cell_chars

    delimiter = _detect_delimiter(text)
    try:
        grid: list[list[str]] = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        logger.warning("Failed to parse CSV %s: %s", source, exc)
        raise CorruptDocumentError(f"Unable to parse CSV: {source}") from exc

    name = file_name or _source_name(source)
    rows: list[TableRow] = build_rows(
        grid,
        file_name=name,
        container="",
        container_label=None,
        max_rows=row_limit,
        max_cell_chars=cell_cap,
    )
    metadata: dict[str, Any] = {"row_count": len(rows), "delimiter": delimiter}
    return ParsedDocument(
        text="\n".join(row.text for row in rows),
        source=source,
        format=_FORMAT,
        title=title,
        metadata=metadata,
        tabular=rows,
    )


def _source_name(source: str) -> str:
    """Base name of *source*, whether it is a path or a URL."""
    if source.startswith(("http://", "https://")):
        return source.rstrip("/").rsplit("/", 1)[-1] or source
    return Path(source).name or source


def _decode(raw: bytes, *, path: str) -> str:
    """Decode CSV bytes, preferring UTF-8 and falling back to Windows-1252."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.warning("CSV %s is not valid UTF-8/CP1252; decoding with replacement.", path)
    try:
        return raw.decode("utf-8", errors="replace")
    except UnicodeDecodeError as exc:  # pragma: no cover - utf-8 with replace never fails
        raise CorruptDocumentError(f"Unable to decode CSV: {path}") from exc


def _detect_delimiter(text: str) -> str:
    """Guess the column separator, preferring the CSV sniffer then heuristics."""
    sample = text[:_SNIFF_CHARS]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_SNIFFER_DELIMITERS).delimiter
    except csv.Error:
        pass
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {delimiter: first_line.count(delimiter) for delimiter in _FALLBACK_DELIMITERS}
    best = max(counts, key=lambda delimiter: counts[delimiter])
    if counts[best]:
        return best
    raise ParsingError(f"Unable to detect the CSV delimiter: {_excerpt(text)}")


def _excerpt(text: str) -> str:
    """Short, safe excerpt used in error messages."""
    return repr(text[:40])
