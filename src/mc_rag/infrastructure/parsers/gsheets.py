"""Google Sheets parser for links shared with *anyone with the link*.

There is no OAuth in this phase: :func:`parse_google_sheets` converts the edit
URL into an export URL (``xlsx`` first, so every worksheet is preserved, then
``csv``) and reuses the regular spreadsheet parsers. A private sheet answers
with a login page or a 401/403, which becomes a clear :class:`ParsingError`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from urllib.parse import urlsplit

from mc_rag.domain.exceptions import ParsingError
from mc_rag.infrastructure.parsers.base import ParsedDocument
from mc_rag.infrastructure.parsers.csv import parse_csv_text
from mc_rag.infrastructure.parsers.url_safety import FetchResult, safe_fetch
from mc_rag.infrastructure.parsers.xlsx import parse_xlsx_bytes

logger = logging.getLogger(__name__)

_FORMAT = "gsheet"
_FILE_NAME = "Google Sheet"
_SHEET_ID_RE = re.compile(r"/spreadsheets/(?:u/\d+/)?d/([A-Za-z0-9_-]{5,})")
_EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format={fmt}"
_FETCHER = Callable[[str], FetchResult]

_PRIVATE_MESSAGE = (
    "This Google Sheet is not shared publicly. Open Share > General access > "
    "'Anyone with the link' and try again."
)


def is_google_sheets_url(url: str) -> bool:
    """Whether *url* points at a Google Sheets document."""
    return bool(_SHEET_ID_RE.search((url or "").strip()))


def parse_google_sheets(
    url: str,
    *,
    fetcher: _FETCHER | None = None,
) -> ParsedDocument:
    """Download a shared Google Sheet and parse it as a spreadsheet.

    Args:
        url: The sheet's edit (or export) URL.
        fetcher: Injected downloader used by tests; defaults to an SSRF-guarded
            HTTP GET.

    Returns:
        The parsed :class:`ParsedDocument`, with ``source`` set to *url*.

    Raises:
        ParsingError: The URL is not a sheet, the sheet is private, or the
            exported file cannot be parsed.
    """
    sheet_id = _sheet_id(url)
    download = fetcher or (lambda target: safe_fetch(target, raise_for_status=False))

    for fmt in ("xlsx", "csv"):
        export_url = _EXPORT_URL.format(sheet_id=sheet_id, fmt=fmt)
        try:
            result = download(export_url)
        except ParsingError:
            raise
        if _looks_private(result):
            logger.warning("Google Sheet %s is not shared publicly.", sheet_id)
            raise ParsingError(_PRIVATE_MESSAGE)
        if result.status_code != 200:
            continue
        document = _parse_export(result, fmt=fmt, original_url=url)
        if document is not None:
            return document
    raise ParsingError(f"Unable to export this Google Sheet: {url}")


def _sheet_id(url: str) -> str:
    """Extract the spreadsheet id from an edit or export URL."""
    match = _SHEET_ID_RE.search((url or "").strip())
    if match is None:
        raise ParsingError("Not a Google Sheets URL (expected docs.google.com/spreadsheets/...).")
    return match.group(1)


def _looks_private(result: FetchResult) -> bool:
    """Whether the export answered with a login/permission page."""
    if result.status_code in (401, 403):
        return True
    host = (urlsplit(result.url).hostname or "").lower()
    if host in {"accounts.google.com", "accounts.google.com.evil.invalid"}:
        return True
    if host.endswith("accounts.google.com"):
        return True
    return result.is_html and b"sign in" in result.body[:4096].lower()


def _parse_export(result: FetchResult, *, fmt: str, original_url: str) -> ParsedDocument | None:
    """Parse one export payload, returning ``None`` when it is unusable."""
    if not result.body:
        return None
    try:
        if fmt == "xlsx":
            document = parse_xlsx_bytes(
                result.body, source=original_url, title=None, file_name=_FILE_NAME
            )
        else:
            document = parse_csv_text(
                _decode(result.body),
                source=original_url,
                title=None,
                file_name=_FILE_NAME,
            )
    except ParsingError as exc:
        logger.debug("Google Sheet export %s unusable: %s", fmt, exc)
        return None
    document.format = _FORMAT
    if not document.title:
        document.title = "Google Sheet"
    return document


def _decode(payload: bytes) -> str:
    """Decode an exported CSV payload (UTF-8 with legacy fallback)."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")
