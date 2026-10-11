"""Package marker for src.mc_rag.infrastructure.parsers.

Convenience re-exports for the public parser API.
"""

from mc_rag.infrastructure.parsers.base import (
    ParsedDocument,
    Section,
    append_to_section,
    coerce_optional_str,
    resolve_document_path,
    slugify,
)
from mc_rag.infrastructure.parsers.docx import parse_docx
from mc_rag.infrastructure.parsers.markdown import parse_markdown
from mc_rag.infrastructure.parsers.pdf import parse_pdf
from mc_rag.infrastructure.parsers.web import parse_html, parse_web

__all__ = [
    "ParsedDocument",
    "Section",
    "append_to_section",
    "coerce_optional_str",
    "parse_docx",
    "parse_html",
    "parse_markdown",
    "parse_pdf",
    "parse_web",
    "resolve_document_path",
    "slugify",
]
