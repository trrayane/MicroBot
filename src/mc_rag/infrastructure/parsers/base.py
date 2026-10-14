"""Shared types and helpers for the document parsers.

Every parser returns a :class:`ParsedDocument` so that downstream stages
(chunking, embedding, indexing) can consume a single normalized shape
regardless of the original file format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mc_rag.domain.exceptions import DocumentNotFoundError, EmptyDocumentError

_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")
_SLUG_SPACE_RE = re.compile(r"[\s_]+")


@dataclass(slots=True)
class Section:
    """A logical section of a parsed document, such as a Markdown heading."""

    level: int
    heading: str
    text: str = ""
    anchor: str | None = None


@dataclass(slots=True, frozen=True)
class TableRow:
    """One row of a tabular document, already rendered as readable text.

    ``text`` carries the file name, the container (sheet or table) and every
    ``header : value`` pair, e.g.
    ``Fichier : bureau.xlsx | Feuille : Bureau | Nom : Amine | Rôle : Trésorier``.
    """

    text: str
    container: str
    row_number: int
    sheet: str = ""


@dataclass(slots=True)
class ParsedDocument:
    """Normalized output produced by every parser."""

    text: str
    source: str
    format: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    # Tabular sources (xlsx/csv) set this to one entry per logical row. The
    # chunker then emits one chunk per row instead of splitting on characters.
    # Invariant: ``text == "\\n".join(row.text for row in tabular)``.
    tabular: list[TableRow] | None = None

    @property
    def is_empty(self) -> bool:
        """Whether the document contains no meaningful text."""
        return not self.text.strip()


def resolve_document_path(file_path: str | Path) -> Path:
    """Validate that ``file_path`` points to a readable, non-empty file.

    Args:
        file_path: Path to the document, absolute or relative.

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        DocumentNotFoundError: The path is missing or is not a regular file.
        EmptyDocumentError: The file exists but is zero bytes.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        raise DocumentNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise DocumentNotFoundError(f"Not a regular file: {path}")
    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"File is empty: {path}")
    return path


def append_to_section(section: Section | None, text: str) -> Section | None:
    """Append *text* to ``section.text`` and return the (unchanged) section.

    Returns ``None`` when there is no active section (i.e. body text appearing
    before the first heading).
    """
    if section is None:
        return None
    section.text = f"{section.text}\n{text}".strip() if section.text else text
    return section


def slugify(text: str) -> str:
    """Build a GitHub-style anchor slug from arbitrary heading text."""
    slug = _SLUG_STRIP_RE.sub("", text.strip().lower())
    return _SLUG_SPACE_RE.sub("-", slug).strip("-")


def coerce_optional_str(value: Any) -> str | None:
    """Return a stripped string for *value*, or ``None`` when it is blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
