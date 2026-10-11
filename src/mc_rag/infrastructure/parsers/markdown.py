"""Markdown document parser built on top of :mod:`markdown_it`.

The public entry point is :func:`parse_markdown`. It returns a normalized
:class:`~mc_rag.infrastructure.parsers.base.ParsedDocument` where:

* ``text`` is the plain-text rendering of the document (headings kept, inline
  markup like ``**bold**`` and ``[links](url)`` resolved to their text);
* ``sections`` holds one entry per heading, carrying the heading level and the
  body text that follows it (useful for structure-aware chunking);
* ``metadata`` records links, heading counts and any YAML front matter.

Unlike :mod:`pdf`, a Markdown source is plain text, so failures are limited to
missing/empty files and unreadable encodings.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from mc_rag.domain.exceptions import EmptyDocumentError, ParsingError
from mc_rag.infrastructure.parsers.base import (
    ParsedDocument,
    Section,
    append_to_section,
    coerce_optional_str,
    resolve_document_path,
    slugify,
)

logger = logging.getLogger(__name__)

_FORMAT = "markdown"
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(?P<body>.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)


def parse_markdown(file_path: str | Path) -> ParsedDocument:
    """Extract text and structure from a Markdown file.

    Args:
        file_path: Path to the ``.md`` document.

    Returns:
        A :class:`ParsedDocument` with the plain text, heading ``sections`` and
        collected links/front-matter metadata.

    Raises:
        DocumentNotFoundError: The path does not exist or is not a file.
        EmptyDocumentError: The file is zero bytes or holds only whitespace.
        ParsingError: The file cannot be read or tokenized.
    """
    path = resolve_document_path(file_path)
    raw = _read_text(path)

    front_matter, body = _split_front_matter(raw)
    if not body.strip():
        raise EmptyDocumentError(f"Markdown file contains no content: {path}")

    try:
        tokens = MarkdownIt("commonmark").parse(body)
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        logger.warning("Failed to parse Markdown %s: %s", path, exc)
        raise ParsingError(f"Unable to parse Markdown file: {path}") from exc

    text, sections, links = _walk_tokens(tokens)
    if not text.strip():
        raise EmptyDocumentError(f"Markdown file contains no extractable text: {path}")

    metadata: dict[str, Any] = {
        "heading_count": len(sections),
        "link_count": len(links),
    }
    if links:
        metadata["links"] = links
    if front_matter:
        metadata["front_matter"] = front_matter

    title = coerce_optional_str(front_matter.get("title"))
    if title is None and sections:
        title = sections[0].heading

    return ParsedDocument(
        text=text,
        source=str(path),
        format=_FORMAT,
        title=title,
        metadata=metadata,
        sections=sections,
    )


def _read_text(path: Path) -> str:
    """Read *path* as UTF-8, falling back to replacement on decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Markdown file %s is not valid UTF-8; decoding with replacement.", path)
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ParsingError(f"Unable to read Markdown file: {path}") from exc


def _walk_tokens(tokens: list[Token]) -> tuple[str, list[Section], list[dict[str, str]]]:
    """Render tokens to plain text, collecting headings and links."""
    lines: list[str] = []
    sections: list[Section] = []
    links: list[dict[str, str]] = []
    current: Section | None = None

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            heading = _inline_to_text(inline, links).strip() if inline is not None else ""
            if heading:
                lines.append(f"{'#' * level} {heading}")
                current = Section(
                    level=level,
                    heading=heading,
                    anchor=slugify(heading),
                )
                sections.append(current)
            index += 2
            continue

        if token.type == "inline":
            inline_text = _inline_to_text(token, links).strip()
            if inline_text:
                lines.append(inline_text)
                current = append_to_section(current, inline_text)
        elif token.type in {"fence", "code_block"}:
            code = token.content.rstrip("\n")
            if code:
                lines.append(code)
                current = append_to_section(current, code)

        index += 1

    return "\n".join(lines).strip(), sections, links


def _inline_to_text(token: Token | None, links: list[dict[str, str]]) -> str:
    """Flatten an inline token's children into plain text, collecting links."""
    if token is None:
        return ""
    if token.children is None:
        return token.content

    parts: list[str] = []
    active_link: dict[str, str] | None = None
    for child in token.children:
        child_type = child.type
        if child_type == "text":
            parts.append(child.content)
            if active_link is not None:
                active_link["text"] += child.content
        elif child_type == "code_inline":
            parts.append(child.content)
        elif child_type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif child_type == "link_open":
            active_link = {"href": _attr_str(child, "href"), "text": ""}
            links.append(active_link)
        elif child_type == "link_close":
            active_link = None
        elif child_type == "image":
            alt = _attr_str(child, "alt") or child.content
            src = _attr_str(child, "src")
            if src:
                links.append({"href": src, "text": alt})
            if alt:
                parts.append(alt)

    return "".join(parts)


def _attr_str(token: Token, name: str) -> str:
    """Return a token attribute coerced to ``str``, or ``""`` when absent."""
    value = token.attrGet(name)
    return "" if value is None else str(value)


def _split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML-style front matter from the document body."""
    match = _FRONT_MATTER_RE.match(raw)
    if match is None:
        return {}, raw
    return _parse_front_matter(match.group("body")), raw[match.end() :]


def _parse_front_matter(block: str) -> dict[str, Any]:
    """Parse simple ``key: value`` front-matter lines (no nested structures)."""
    data: dict[str, Any] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        data[key.strip()] = value.strip().strip("'\"")
    return data
