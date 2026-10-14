"""Excel ``.xlsx`` parser built on :mod:`openpyxl`.

The public entry point is :func:`parse_xlsx`. Every worksheet becomes a
:class:`~mc_rag.infrastructure.parsers.base.TableRow` per data row, rendered
with its headers, its file name and its sheet name, so the chunker can keep one
row = one chunk.

Behaviour worth knowing:

* formulas are read through their **cached** result (``data_only=True``); a
  workbook that was never opened by Excel stores no result, so those cells read
  as empty (see EDGE_CASES.md);
* merged cells propagate the top-left value to the whole merged range;
* empty rows/columns are dropped, the header row is auto-detected;
* ``SHEET_MAX_ROWS`` bounds the work and raises a clear error.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import CorruptDocumentError, EmptyDocumentError
from mc_rag.infrastructure.parsers.base import ParsedDocument, TableRow, resolve_document_path
from mc_rag.infrastructure.parsers.table import SHEET_LABEL, build_rows

logger = logging.getLogger(__name__)

_FORMAT = "xlsx"


def parse_xlsx(
    file_path: str | Path,
    *,
    max_rows: int | None = None,
    max_cell_chars: int | None = None,
) -> ParsedDocument:
    """Extract every worksheet of an Excel workbook as rendered rows.

    Args:
        file_path: Path to the ``.xlsx`` workbook.
        max_rows: Row budget per workbook (defaults to ``SHEET_MAX_ROWS``).
        max_cell_chars: Per-cell character cap (defaults to ``SHEET_MAX_CELL_CHARS``).

    Returns:
        A :class:`ParsedDocument` whose ``tabular`` holds one entry per data row.

    Raises:
        DocumentNotFoundError: The path does not exist or is not a file.
        EmptyDocumentError: The file is zero bytes or no sheet holds a value.
        CorruptDocumentError: The file is not a readable Excel workbook.
        DocumentTooLargeError: The workbook has more rows than *max_rows*.
    """
    path = resolve_document_path(file_path)
    try:
        with path.open("rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise CorruptDocumentError(f"Unable to read Excel file: {path}") from exc
    return parse_xlsx_bytes(
        payload, source=str(path), title=path.stem, max_rows=max_rows,
        max_cell_chars=max_cell_chars,
    )


def parse_xlsx_bytes(
    payload: bytes,
    *,
    source: str,
    title: str | None = None,
    max_rows: int | None = None,
    max_cell_chars: int | None = None,
    file_name: str | None = None,
) -> ParsedDocument:
    """Parse an in-memory Excel workbook (used for uploads and Google Sheets).

    Args:
        payload: Raw ``.xlsx`` bytes.
        source: Provenance recorded on the document (path or URL).
        title: Document title, falling back to the workbook's own title.
        max_rows: Row budget (defaults to ``SHEET_MAX_ROWS``).
        max_cell_chars: Per-cell character cap.
        file_name: Name rendered inside every row; defaults to the ``source``
            base name.
    """
    if not payload:
        raise EmptyDocumentError(f"Excel file is empty: {source}")
    try:
        workbook = load_workbook(io.BytesIO(payload), data_only=True)
    except (InvalidFileException, ValueError, KeyError) as exc:
        logger.warning("Failed to open Excel %s: %s", source, exc)
        raise CorruptDocumentError(f"Unable to open Excel workbook: {source}") from exc
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        logger.warning("Failed to open Excel %s: %s", source, exc)
        raise CorruptDocumentError(f"Unable to open Excel workbook: {source}") from exc

    try:
        return _workbook_to_document(
            workbook, source=source, title=title, max_rows=max_rows,
            max_cell_chars=max_cell_chars, file_name=file_name,
        )
    finally:
        workbook.close()


def _workbook_to_document(
    workbook: Workbook,
    *,
    source: str,
    title: str | None,
    max_rows: int | None,
    max_cell_chars: int | None,
    file_name: str | None = None,
) -> ParsedDocument:
    """Render every worksheet into rows and assemble the parsed document."""
    settings = get_settings()
    row_limit = max_rows if max_rows is not None else settings.sheet_max_rows
    cell_cap = max_cell_chars if max_cell_chars is not None else settings.sheet_max_cell_chars

    name = file_name or _file_name(source)
    sheet_names: list[str] = []
    all_rows: list[TableRow] = []
    for worksheet in _worksheets(workbook):
        try:
            rows = build_rows(
                _matrix(worksheet),
                file_name=name,
                container=worksheet.title,
                container_label=SHEET_LABEL,
                sheet=worksheet.title,
                max_rows=row_limit,
                max_cell_chars=cell_cap,
            )
        except EmptyDocumentError:
            logger.debug("Skipping empty worksheet %r of %s", worksheet.title, source)
            continue
        sheet_names.append(worksheet.title)
        all_rows.extend(rows)

    if not all_rows:
        raise EmptyDocumentError(f"Excel workbook contains no value: {source}")

    metadata: dict[str, Any] = {
        "sheet_count": len(sheet_names),
        "sheets": sheet_names,
        "row_count": len(all_rows),
    }
    return ParsedDocument(
        text="\n".join(row.text for row in all_rows),
        source=source,
        format=_FORMAT,
        title=title or workbook.properties.title or None,
        metadata=metadata,
        tabular=all_rows,
    )


def _worksheets(workbook: Workbook) -> Iterable[Worksheet]:
    """Every worksheet of *workbook*, skipping chart-only sheets."""
    return [sheet for sheet in workbook.worksheets if isinstance(sheet, Worksheet)]


def _matrix(worksheet: Worksheet) -> list[list[object]]:
    """Read the sheet values, extending merged cells vertically.

    A merged range repeats its top value **down each column of the range**: a
    vertically merged label is therefore read on every covered row, while a
    horizontally merged cell does not duplicate its value into the next column
    (that column simply reads empty for the row, and gets dropped when the row
    is rendered).
    """
    grid: list[list[object]] = [
        [cell.value for cell in row] for row in worksheet.iter_rows()
    ]
    for merged in worksheet.merged_cells.ranges:
        start_row = merged.min_row - 1
        if start_row >= len(grid):
            continue
        for column_index in range(merged.min_col - 1, merged.max_col):
            if column_index >= len(grid[start_row]):
                continue
            value = grid[start_row][column_index]
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            for row_index in range(start_row + 1, merged.max_row):
                if row_index < len(grid) and column_index < len(grid[row_index]):
                    grid[row_index][column_index] = value
    return grid


def _file_name(source: str) -> str:
    """File name repeated in the rendered rows."""
    if source.startswith(("http://", "https://")):
        return source.rsplit("/", 1)[-1] or source
    return Path(source).name or source
