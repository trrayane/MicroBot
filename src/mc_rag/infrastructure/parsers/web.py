"""Web / HTML document parser built on :mod:`httpx` and :mod:`bs4`.

The public entry points are:

* :func:`parse_web` - accepts an HTTP(S) URL, a raw HTML string, or a path to a
  local ``.html`` file and dispatches accordingly.
* :func:`parse_html` - parses an HTML string directly (useful for tests and for
  callers that have already fetched the markup).

Scripts, styles and chrome (navigation, footers, forms, ...) are stripped before
the readable text is extracted. Headings become
:class:`~mc_rag.infrastructure.parsers.base.Section` entries and the page title,
meta description and language are surfaced in the metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment
from bs4.element import Tag

from mc_rag.domain.exceptions import EmptyDocumentError, ParsingError
from mc_rag.infrastructure.parsers.base import (
    ParsedDocument,
    Section,
    append_to_section,
    coerce_optional_str,
    resolve_document_path,
    slugify,
)
from mc_rag.infrastructure.parsers.url_safety import safe_fetch

logger = logging.getLogger(__name__)

_FORMAT = "web"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_USER_AGENT = "mc-club-rag/0.1 (+https://github.com/mc-club-rag)"
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = (*_HEADING_TAGS, "p", "li", "pre", "blockquote", "td")
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
)


def parse_web(
    url_or_html: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    headers: Mapping[str, str] | None = None,
    allow_local_files: bool = True,
) -> ParsedDocument:
    """Parse a URL, a raw HTML string, or a local HTML file.

    Args:
        url_or_html: An ``http(s)://`` URL, an HTML string, or a filesystem path.
        timeout: Request timeout in seconds (only used for URLs).
        headers: Extra HTTP headers merged over the default user agent.
        allow_local_files: Whether a filesystem path is accepted. The URL upload
            endpoint passes ``False`` so a submitted "URL" cannot read files
            from the host.

    Returns:
        The parsed :class:`ParsedDocument`.

    Raises:
        EmptyDocumentError: The input, response body or extracted text is empty.
        UnsafeURLError: The URL scheme, host or resolved address is refused
            (SSRF guard, see :mod:`mc_rag.infrastructure.parsers.url_safety`).
        ParsingError: The input is unrecognized, the fetch fails, or the file
            cannot be read.
    """
    candidate = url_or_html.strip()
    if not candidate:
        raise EmptyDocumentError("Empty URL or HTML input.")

    if _looks_like_url(candidate):
        html = _fetch(candidate, timeout=timeout, headers=headers)
        return parse_html(html, source=candidate)

    if "<" in candidate:
        return parse_html(url_or_html, source="<string>")

    path = Path(candidate).expanduser()
    if allow_local_files and path.is_file():
        resolved = resolve_document_path(path)
        return parse_html(_read_html_file(resolved), source=str(resolved))

    raise ParsingError(
        "Input is neither an HTTP(S) URL, an HTML string, nor an existing file: "
        f"{candidate}"
    )


def parse_html(html: str, *, source: str = "<string>") -> ParsedDocument:
    """Extract readable text and structure from an HTML string.

    Args:
        html: The raw HTML markup.
        source: Identifier recorded on the result (URL, path or ``"<string>"``).

    Returns:
        The parsed :class:`ParsedDocument`.

    Raises:
        EmptyDocumentError: The markup or its extracted text is empty.
        ParsingError: The markup cannot be parsed at all.
    """
    if not html or not html.strip():
        raise EmptyDocumentError(f"Empty HTML input from: {source}")

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        logger.warning("Failed to parse HTML from %s: %s", source, exc)
        raise ParsingError(f"Unable to parse HTML from: {source}") from exc

    title = _extract_title(soup)
    _strip_noise(soup)
    text, sections, headings = _extract_content(soup)

    if not text.strip():
        raise EmptyDocumentError(f"HTML contains no extractable text: {source}")

    metadata: dict[str, Any] = {"heading_count": len(sections)}
    description = _extract_meta(soup, "description")
    if description:
        metadata["description"] = description
    language = _extract_language(soup)
    if language:
        metadata["language"] = language
    if headings:
        metadata["headings"] = headings

    return ParsedDocument(
        text=text,
        source=source,
        format=_FORMAT,
        title=title,
        metadata=metadata,
        sections=sections,
    )


def _looks_like_url(value: str) -> bool:
    """Whether *value* starts with an ``http``/``https`` scheme."""
    return value.lower().startswith(("http://", "https://"))


def _fetch(url: str, *, timeout: float, headers: Mapping[str, str] | None) -> str:
    """Fetch *url* through the SSRF guard and return its decoded body.

    Raises:
        UnsafeURLError: The scheme, host or resolved address is refused.
        EmptyDocumentError: The response body is empty.
        ParsingError: The HTTP request failed.
    """
    request_headers = {"User-Agent": _DEFAULT_USER_AGENT}
    if headers:
        request_headers.update(headers)

    result = safe_fetch(url, timeout=timeout, headers=request_headers)
    body = result.body.decode(_charset(result.content_type), errors="replace")
    if not body or not body.strip():
        raise EmptyDocumentError(f"Empty response body from: {url}")
    return body


def _charset(content_type: str) -> str:
    """Extract the charset of a ``Content-Type`` header, defaulting to UTF-8."""
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"'")
    return "utf-8"


def _read_html_file(path: Path) -> str:
    """Read a local HTML file as UTF-8, falling back to replacement decoding."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("HTML file %s is not valid UTF-8; decoding with replacement.", path)
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ParsingError(f"Unable to read HTML file: {path}") from exc


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Return the page ``<title>``, falling back to the first heading."""
    if soup.title is not None:
        title = soup.title.get_text(strip=True)
        if title:
            return title
    first_heading = soup.find(_HEADING_TAGS)
    if isinstance(first_heading, Tag):
        return coerce_optional_str(first_heading.get_text(strip=True))
    return None


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove comments and non-content elements from the tree in place."""
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for name in _NOISE_TAGS:
        for tag in soup.find_all(name):
            if isinstance(tag, Tag):
                tag.decompose()


def _extract_content(
    soup: BeautifulSoup,
) -> tuple[str, list[Section], list[dict[str, Any]]]:
    """Walk block-level elements into text, collecting headings and sections."""
    lines: list[str] = []
    sections: list[Section] = []
    headings: list[dict[str, Any]] = []
    current: Section | None = None

    for element in soup.find_all(_BLOCK_TAGS):
        if not isinstance(element, Tag):
            continue
        if element.find_parent(_BLOCK_TAGS) is not None:
            continue

        text = _element_text(element)
        if not text:
            continue

        level = _heading_level(element.name)
        if level is None:
            lines.append(text)
            current = append_to_section(current, text)
            continue

        lines.append(f"{'#' * level} {text}")
        anchor = _attr_str(element, "id") or slugify(text)
        current = Section(level=level, heading=text, anchor=anchor)
        sections.append(current)
        headings.append({"level": level, "text": text})

    return "\n".join(lines).strip(), sections, headings


def _element_text(element: Tag) -> str:
    """Return the collapsed text of *element*, preserving ``<pre>`` whitespace."""
    if element.name.lower() == "pre":
        return element.get_text().strip()
    return " ".join(element.get_text(" ", strip=True).split())


def _heading_level(tag_name: str) -> int | None:
    """Return the numeric level for an ``h1``-``h6`` tag, else ``None``."""
    name = tag_name.lower()
    if name in _HEADING_TAGS:
        return int(name[1])
    return None


def _extract_meta(soup: BeautifulSoup, name: str) -> str | None:
    """Return the ``content`` of ``<meta name="...">`` when present."""
    tag = soup.find("meta", attrs={"name": name})
    if isinstance(tag, Tag):
        return coerce_optional_str(_attr_str(tag, "content"))
    return None


def _extract_language(soup: BeautifulSoup) -> str | None:
    """Return the ``lang`` attribute of the root ``<html>`` element."""
    html = soup.find("html")
    if isinstance(html, Tag):
        return coerce_optional_str(_attr_str(html, "lang"))
    return None


def _attr_str(tag: Tag, name: str) -> str:
    """Return an attribute coerced to ``str``, or ``""`` when absent."""
    value = tag.get(name)
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
