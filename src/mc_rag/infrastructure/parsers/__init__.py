"""Package marker for src.mc_rag.infrastructure.parsers.

Convenience re-exports for the public parser API.
"""

from mc_rag.infrastructure.parsers.base import (
    ParsedDocument,
    Section,
    TableRow,
    append_to_section,
    coerce_optional_str,
    resolve_document_path,
    slugify,
)
from mc_rag.infrastructure.parsers.csv import parse_csv, parse_csv_text
from mc_rag.infrastructure.parsers.docx import parse_docx
from mc_rag.infrastructure.parsers.gsheets import is_google_sheets_url, parse_google_sheets
from mc_rag.infrastructure.parsers.markdown import parse_markdown
from mc_rag.infrastructure.parsers.pdf import parse_pdf
from mc_rag.infrastructure.parsers.web import parse_html, parse_web
from mc_rag.infrastructure.parsers.xlsx import parse_xlsx

__all__ = [
    "ParsedDocument",
    "Section",
    "TableRow",
    "append_to_section",
    "coerce_optional_str",
    "is_google_sheets_url",
    "parse_csv",
    "parse_csv_text",
    "parse_docx",
    "parse_google_sheets",
    "parse_html",
    "parse_markdown",
    "parse_pdf",
    "parse_web",
    "parse_xlsx",
    "resolve_document_path",
    "slugify",
]
