"""Document chunking strategies.

The chunker converts a :class:`~mc_rag.infrastructure.parsers.base.ParsedDocument`
into a list of :class:`Chunk` objects ready for embedding and indexing.

Splitting strategy
------------------
Splitting is **recursive and structure-aware**, operating on characters with a
separator hierarchy that prefers the largest natural boundary available::

    paragraph -> line -> sentence -> clause -> word -> character

Text is first split on ``"\\n\\n"``; any resulting piece that is still larger than
``chunk_size`` is recursively re-split on the next separator, and so on. This
keeps sentences and words intact whenever the size budget allows, and only falls
back to a hard character cut when a single token exceeds ``chunk_size``.
Separators are kept attached to their preceding piece, so concatenating all
pieces reproduces the source exactly. Adjacent pieces are then packed into
chunks of at most ``chunk_size`` characters, with an approximate
``chunk_overlap`` of trailing context re-used between neighbours (trimmed to a
word boundary).

Defaults for ``chunk_size``/``chunk_overlap`` come from
:mod:`mc_rag.config` (``CHUNK_SIZE`` / ``CHUNK_OVERLAP``).
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from mc_rag.config import get_settings
from mc_rag.infrastructure.parsers.base import ParsedDocument, slugify

_settings = get_settings()

DEFAULT_CHUNK_SIZE: int = _settings.chunk_size
DEFAULT_CHUNK_OVERLAP: int = _settings.chunk_overlap

_DEFAULT_VISIBILITY = "public"
_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)
_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?P<hashes>#{1,6})[ \t]+(?P<text>.+?)[ \t]*#*[ \t]*$",
    re.MULTILINE,
)


@dataclass(slots=True)
class Chunk:
    """A single embeddable unit of text with provenance metadata."""

    text: str
    chunk_id: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk's text."""
        return len(self.text)


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    visibility: str | None = None,
) -> list[Chunk]:
    """Split *document* into overlapping, structure-aware chunks.

    Args:
        document: The parsed document to split.
        chunk_size: Maximum number of characters per chunk (must be positive).
        chunk_overlap: Characters of trailing context shared with the next chunk
            (must be non-negative and smaller than ``chunk_size``).
        visibility: Overrides the document metadata ``visibility`` value; falls
            back to the document metadata and then to ``"public"``.

    Returns:
        A list of :class:`Chunk` objects in document order, or an empty list
        when the document has no extractable text.

    Raises:
        ValueError: If ``chunk_size``/``chunk_overlap`` are invalid.
    """
    _validate_params(chunk_size, chunk_overlap)

    text = document.text
    if not text or not text.strip():
        return []

    resolved_visibility = _resolve_visibility(document, visibility)
    pieces = _recursive_split(text, chunk_size, _SEPARATORS)
    spans = _merge_pieces(pieces, chunk_size, chunk_overlap)
    if not spans:
        return []

    headings = _heading_offsets(text)
    heading_offsets = [offset for offset, _, _ in headings]
    chunk_count = len(spans)

    chunks: list[Chunk] = []
    for index, (chunk_text, start, end) in enumerate(spans):
        heading, level = _section_for(headings, heading_offsets, start)
        chunk_id = _make_chunk_id(document.source, index, chunk_text)
        metadata: dict[str, Any] = {
            "chunk_id": chunk_id,
            "source": document.source,
            "format": document.format,
            "title": document.title,
            "visibility": resolved_visibility,
            "section_heading": heading,
            "section_anchor": slugify(heading) if heading else None,
            "section_level": level,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "start_char": start,
            "end_char": end,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "document_metadata": dict(document.metadata),
        }
        chunks.append(
            Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                index=index,
                start_char=start,
                end_char=end,
                metadata=metadata,
            )
        )
    return chunks


def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk sizing parameters, raising ``ValueError`` when invalid."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _resolve_visibility(document: ParsedDocument, visibility: str | None) -> str:
    """Resolve the effective visibility from the argument or document metadata."""
    candidate = visibility if visibility is not None else document.metadata.get("visibility")
    if candidate is None:
        return _DEFAULT_VISIBILITY
    text = str(candidate).strip()
    return text or _DEFAULT_VISIBILITY


def _recursive_split(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively split *text* into pieces no larger than *chunk_size*."""
    if len(text) <= chunk_size:
        return [text] if text else []

    separator = separators[-1]
    remaining: tuple[str, ...] = ()
    for index, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            break
        if candidate in text:
            separator = candidate
            remaining = separators[index + 1 :]
            break

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    for part in _split_keeping(text, separator):
        if not part:
            continue
        if len(part) <= chunk_size:
            pieces.append(part)
        elif remaining:
            pieces.extend(_recursive_split(part, chunk_size, remaining))
        else:
            pieces.extend(part[i : i + chunk_size] for i in range(0, len(part), chunk_size))
    return pieces


def _split_keeping(text: str, separator: str) -> list[str]:
    """Split on *separator* while keeping it attached to each preceding piece."""
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    pieces.append(parts[-1])
    return pieces


def _merge_pieces(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[tuple[str, int, int]]:
    """Pack *pieces* into ``(text, start_char, end_char)`` spans with overlap."""
    spans: list[tuple[str, int, int]] = []
    current = ""
    current_start = 0
    offset = 0

    for piece in pieces:
        piece_start = offset
        offset += len(piece)
        if not piece:
            continue
        if not current:
            current = piece
            current_start = piece_start
        elif len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            _append_span(spans, current, current_start)
            seed, seed_start = _overlap_seed(
                current, current_start, piece, chunk_size, chunk_overlap
            )
            current = seed + piece
            current_start = seed_start

    _append_span(spans, current, current_start)
    return spans


def _append_span(spans: list[tuple[str, int, int]], text: str, start: int) -> None:
    """Append a stripped ``(text, start, end)`` span when it has content."""
    stripped = text.strip()
    if not stripped:
        return
    leading = len(text) - len(text.lstrip())
    trailing = len(text) - len(text.rstrip())
    spans.append((stripped, start + leading, start + len(text) - trailing))


def _overlap_seed(
    previous: str,
    previous_start: int,
    piece: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, int]:
    """Return the ``(seed, start)`` trailing context to prepend to the next chunk."""
    if chunk_overlap <= 0 or not previous:
        return "", previous_start + len(previous)

    seed_length = min(chunk_overlap, len(previous))
    seed = previous[-seed_length:]
    seed_start = previous_start + len(previous) - seed_length

    space_index = seed.find(" ")
    if space_index != -1:
        seed = seed[space_index + 1 :]
        seed_start += space_index + 1

    max_seed = max(0, chunk_size - len(piece))
    if len(seed) > max_seed:
        excess = len(seed) - max_seed
        seed = seed[excess:]
        seed_start += excess

    return seed, seed_start


def _heading_offsets(text: str) -> list[tuple[int, str, int]]:
    """Return ``(offset, heading_text, level)`` for every Markdown heading line."""
    headings: list[tuple[int, str, int]] = []
    for match in _HEADING_RE.finditer(text):
        heading = match.group("text").strip()
        if heading:
            headings.append((match.start(), heading, len(match.group("hashes"))))
    return headings


def _section_for(
    headings: list[tuple[int, str, int]],
    heading_offsets: list[int],
    start: int,
) -> tuple[str | None, int | None]:
    """Return the section heading/level that was active at character *start*."""
    position = bisect_right(heading_offsets, start) - 1
    if position < 0:
        return None, None
    _, heading, level = headings[position]
    return heading, level


def _make_chunk_id(source: str, index: int, text: str) -> str:
    """Build a deterministic, content-addressed id for a chunk."""
    digest = hashlib.sha1(f"{source}|{index}|{text}".encode()).hexdigest()
    return digest[:16]
