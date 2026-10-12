"""Builds the ``sources`` returned with an answer from the chunks the model cited."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

from mc_rag.infrastructure.vector_store.chroma_store import SearchResult

DEFAULT_MAX_SNIPPET = 200
DEFAULT_FALLBACK_LIMIT = 3

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACES_RE = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class Source:
    """One cited document, as exposed by the API."""

    doc_id: str
    title: str
    snippet: str


def build_sources(
    answer: str,
    hits: Sequence[SearchResult],
    *,
    max_snippet: int = DEFAULT_MAX_SNIPPET,
    fallback_limit: int = DEFAULT_FALLBACK_LIMIT,
) -> list[Source]:
    """Return one :class:`Source` per document cited in *answer*.

    Citations are ``[n]`` markers pointing at *hits* (1-based). Markers that do
    not match a retrieved chunk are ignored. When the answer cites nothing valid,
    the top retrieved documents are returned instead (at most *fallback_limit*).
    """
    cited = _cited_hits(answer, hits)
    if not cited:
        cited = list(hits)
        limit = fallback_limit
    else:
        limit = len(cited)

    sources: list[Source] = []
    seen: set[str] = set()
    for hit in cited:
        doc_id = str(hit.metadata.get("doc_id") or hit.chunk_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append(
            Source(doc_id=doc_id, title=_title(hit), snippet=_snippet(hit.text, max_snippet))
        )
        if len(sources) >= limit:
            break
    return sources


def _cited_hits(answer: str, hits: Sequence[SearchResult]) -> list[SearchResult]:
    ordered: list[SearchResult] = []
    for group in _MARKER_RE.findall(answer):
        for raw in group.split(","):
            index = int(raw.strip())
            if 1 <= index <= len(hits):
                ordered.append(hits[index - 1])
    return ordered


def _title(hit: SearchResult) -> str:
    title = hit.metadata.get("title")
    if title:
        return str(title)
    source = str(hit.metadata.get("source") or "")
    return PurePath(source).stem if source else "document"


def _snippet(text: str, limit: int) -> str:
    collapsed = _SPACES_RE.sub(" ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 1, 0)].rstrip() + "…"