"""Shared helpers for tabular sources (Excel, CSV, DOCX tables).

Every tabular parser converts its rows through :func:`build_rows`, which

* drops empty rows and columns,
* detects the header row (or generates ``Colonne N`` labels when absent),
* formats cell values readably (dates, numbers, booleans, merged cells),
* renders one self-contained line per row::

      Fichier : bureau.xlsx | Feuille : Bureau | Nom : Amine | Rôle : Trésorier

The rendered line is what the chunker later turns into exactly one chunk, so a
row is never split on characters.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import DocumentTooLargeError, EmptyDocumentError
from mc_rag.infrastructure.parsers.base import TableRow

FILE_LABEL = "Fichier"
SHEET_LABEL = "Feuille"
TABLE_LABEL = "Tableau"
GENERATED_HEADER_PREFIX = "Colonne"
DEFAULT_MAX_ROWS = get_settings().sheet_max_rows
DEFAULT_MAX_CELL_CHARS = get_settings().sheet_max_cell_chars
# How far down (in non-empty rows) we look for the real header row.
HEADER_SCAN_ROWS = 25

_ELLIPSIS = "…"


def format_cell(value: object, *, max_chars: int = DEFAULT_MAX_CELL_CHARS) -> str:
    """Render one cell as a short, readable string.

    Dates use ISO ``YYYY-MM-DD`` (plus ``HH:MM`` when a time is present) so the
    text stays unambiguous in French, English and Arabic documents alike.
    Floats that hold an integer lose their trailing ``.0``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, _dt.datetime):
        stamp = value.strftime("%Y-%m-%d %H:%M") if value.time() != _dt.time() else value.strftime(
            "%Y-%m-%d"
        )
        return _truncate(stamp, max_chars)
    if isinstance(value, _dt.date):
        return _truncate(value.strftime("%Y-%m-%d"), max_chars)
    if isinstance(value, _dt.time):
        return _truncate(value.strftime("%H:%M"), max_chars)
    if isinstance(value, float) and value.is_integer():
        return _truncate(str(int(value)), max_chars)
    return _truncate(_collapse(str(value)), max_chars)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace and strip the ends."""
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    """Shorten *text* to *max_chars* characters (0 or less disables the cap)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(max_chars - 1, 1)].rstrip() + _ELLIPSIS


def _is_blank(value: object) -> bool:
    """Whether a cell holds no information."""
    return value is None or (isinstance(value, str) and not value.strip())


def looks_like_header(row: Sequence[object]) -> bool:
    """Whether *row* can be the header row.

    A row is a header when every non-empty cell is already text: numbers,
    dates and booleans are values, not column names.
    """
    filled = [cell for cell in row if not _is_blank(cell)]
    if not filled:
        return False
    return all(isinstance(cell, str) for cell in filled)


def find_header_row(
    rows: Sequence[Sequence[object]],
    non_empty: Sequence[int],
    *,
    scan: int = HEADER_SCAN_ROWS,
) -> int | None:
    """Index of the header row: the widest all-text row near the top.

    Real spreadsheets often start with title banners (``Université…``, a
    version stamp, a merged ``Présences`` band…) before the actual header, so
    taking the first non-empty row would label every column ``Colonne N`` and
    index the banner as data. The header is instead the all-text row with the
    most filled cells among the first *scan* non-empty rows; ties are resolved
    towards the top. Returns ``None`` when no row looks like a header.
    """
    best_index: int | None = None
    best_filled = 0
    for index in non_empty[:scan]:
        row = rows[index]
        if not looks_like_header(row):
            continue
        filled = sum(1 for cell in row if not _is_blank(cell))
        if filled > best_filled:
            best_index, best_filled = index, filled
    return best_index


def _header_labels(row: Sequence[object], width: int) -> list[str]:
    """Build unique, non-empty header labels for *row* padded to *width*."""
    labels: list[str] = []
    seen: dict[str, int] = {}
    for index in range(width):
        raw = _collapse(str(row[index])) if index < len(row) and not _is_blank(row[index]) else ""
        label = raw or f"{GENERATED_HEADER_PREFIX} {index + 1}"
        count = seen.get(label, 0) + 1
        seen[label] = count
        labels.append(label if count == 1 else f"{label} ({count})")
    return labels


def build_rows(
    rows: Sequence[Sequence[object]],
    *,
    file_name: str,
    container: str,
    container_label: str | None = None,
    sheet: str = "",
    max_rows: int | None = None,
    max_cell_chars: int | None = None,
) -> list[TableRow]:
    """Convert a grid of cell values into rendered :class:`TableRow` objects.

    Args:
        rows: Raw rows as read from the source (values may be of any type).
        file_name: File name repeated in every rendered row.
        container: Sheet name or table index (ignored when *container_label*
            is ``None``).
        container_label: Label before *container* (``Feuille``, ``Tableau``).
        sheet: Sheet name recorded on each row ("" for DOCX tables).
        max_rows: Row budget; raises when exceeded. Defaults to ``SHEET_MAX_ROWS``.
        max_cell_chars: Per-cell cap; 0 disables it.

    Raises:
        EmptyDocumentError: No row holds any value.
        DocumentTooLargeError: More rows than *max_rows*.
    """
    limit = max_rows if max_rows is not None else DEFAULT_MAX_ROWS
    cell_cap = max_cell_chars if max_cell_chars is not None else DEFAULT_MAX_CELL_CHARS

    non_empty = [index for index, row in enumerate(rows) if any(not _is_blank(c) for c in row)]
    if not non_empty:
        raise EmptyDocumentError(f"Spreadsheet contains no value: {file_name}")

    header_index = find_header_row(rows, non_empty)
    # Rows above the header are title banners, not data.
    if header_index is not None:
        data_indexes = [index for index in non_empty if index > header_index]
    else:
        data_indexes = non_empty

    width = max((len(rows[index]) for index in non_empty), default=0)
    if width == 0:
        raise EmptyDocumentError(f"Spreadsheet contains no value: {file_name}")

    if header_index is not None:
        headers = _header_labels(rows[header_index], width)
    else:
        headers = _header_labels([], width)
    active = _active_columns(
        rows,
        data_indexes,
        width,
        header_index=header_index,
    )

    rendered: list[TableRow] = []
    for index in data_indexes:
        row = rows[index]
        values = [
            format_cell(row[column] if column < len(row) else None, max_chars=cell_cap)
            for column in active
        ]
        if not any(values):
            continue
        text = render_row(
            file_name,
            container if container_label else None,
            container_label,
            [headers[column] for column in active],
            values,
        )
        if not text:
            continue
        if len(rendered) >= limit:
            raise DocumentTooLargeError(
                f"Spreadsheet has more than {limit} rows ({file_name}); "
                "raise SHEET_MAX_ROWS to ingest it."
            )
        rendered.append(TableRow(text=text, container=container, row_number=index + 1, sheet=sheet))

    if not rendered:
        raise EmptyDocumentError(f"Spreadsheet contains no data row: {file_name}")
    return rendered


def _active_columns(
    rows: Sequence[Sequence[object]],
    data_indexes: Sequence[int],
    width: int,
    *,
    header_index: int | None,
) -> list[int]:
    """Columns that hold a header or at least one value."""
    active: list[int] = []
    header = rows[header_index] if header_index is not None else ()
    for column in range(width):
        if column < len(header) and not _is_blank(header[column]):
            active.append(column)
            continue
        if any(
            column < len(rows[index]) and not _is_blank(rows[index][column])
            for index in data_indexes
        ):
            active.append(column)
    return active


def render_row(
    file_name: str,
    container: str | None,
    container_label: str | None,
    headers: Sequence[str],
    values: Sequence[str],
) -> str:
    """Join ``label : value`` pairs into a single readable line."""
    parts = [f"{FILE_LABEL} : {file_name}"]
    if container is not None and container_label:
        parts.append(f"{container_label} : {container}")
    for header, value in zip(headers, values, strict=False):
        if not value:
            continue
        parts.append(f"{header} : {value}")
    if len(parts) == 1 or (len(parts) == 2 and container is not None):
        return ""
    return " | ".join(parts)
