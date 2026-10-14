"""Unit tests for :mod:`mc_rag.rag.chunking`."""

from __future__ import annotations

from typing import Any

import pytest

from mc_rag.config import get_settings
from mc_rag.infrastructure.parsers.base import ParsedDocument, Section, TableRow
from mc_rag.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    chunk_document,
)


def make_document(
    text: str,
    *,
    sections: list[Section] | None = None,
    metadata: dict[str, Any] | None = None,
    title: str | None = "Doc Title",
    source: str = "doc.md",
    fmt: str = "markdown",
) -> ParsedDocument:
    """Build a ParsedDocument fixture for chunking tests."""
    return ParsedDocument(
        text=text,
        source=source,
        format=fmt,
        title=title,
        metadata=metadata if metadata is not None else {},
        sections=sections if sections is not None else [],
    )


def paragraphs(
    count: int, sentence: str = "This is a reasonably long sentence about the club."
) -> str:
    """Return *count* blank-line separated paragraphs."""
    return "\n\n".join(f"{sentence} Number {i}." for i in range(count))


def test_defaults_come_from_settings() -> None:
    settings = get_settings()
    assert DEFAULT_CHUNK_SIZE == settings.chunk_size
    assert DEFAULT_CHUNK_OVERLAP == settings.chunk_overlap
    assert DEFAULT_CHUNK_SIZE > 0
    assert 0 <= DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


def test_short_document_yields_single_chunk() -> None:
    doc = make_document("A short note.")
    chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "A short note."
    assert chunks[0].index == 0
    assert chunks[0].metadata["chunk_count"] == 1
    assert chunks[0].char_count <= 512


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t \n "])
def test_empty_documents_return_no_chunks(text: str) -> None:
    assert chunk_document(make_document(text)) == []


def test_long_document_is_split_respecting_size() -> None:
    doc = make_document(paragraphs(30))
    chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.char_count <= 120 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["chunk_count"] == len(chunks) for chunk in chunks)


def test_chunks_are_exact_substrings_of_source() -> None:
    text = paragraphs(20)
    doc = make_document(text)
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_words_are_not_cut_in_half() -> None:
    text = paragraphs(20, sentence="alpha bravo charlie delta echo foxtrot")
    doc = make_document(text)
    vocabulary = set(text.split())
    chunks = chunk_document(doc, chunk_size=90, chunk_overlap=15)
    assert len(chunks) > 1
    for chunk in chunks:
        assert set(chunk.text.split()).issubset(vocabulary)


def test_overlap_creates_shared_context() -> None:
    doc = make_document(paragraphs(20))
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=40)
    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start_char < previous.end_char


def test_zero_overlap_produces_disjoint_ranges() -> None:
    doc = make_document(paragraphs(20))
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=0)
    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start_char >= previous.end_char


def test_section_tracking_across_chunks() -> None:
    text = "# Alpha\n" + paragraphs(10) + "\n\n# Beta\n" + paragraphs(10)
    sections = [
        Section(level=1, heading="Alpha", anchor="alpha"),
        Section(level=1, heading="Beta", anchor="beta"),
    ]
    doc = make_document(text, sections=sections)
    chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)

    alpha_chunks = [c for c in chunks if c.metadata["section_heading"] == "Alpha"]
    beta_chunks = [c for c in chunks if c.metadata["section_heading"] == "Beta"]
    assert alpha_chunks and beta_chunks
    assert all(c.metadata["section_level"] == 1 for c in chunks)
    assert alpha_chunks[0].metadata["section_anchor"] == "alpha"
    assert beta_chunks[0].metadata["section_anchor"] == "beta"
    assert max(c.start_char for c in alpha_chunks) < min(c.start_char for c in beta_chunks)


def test_missing_sections_is_handled() -> None:
    doc = make_document("Plain body without any headings, repeated. " * 20)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    assert chunks
    assert all(chunk.metadata["section_heading"] is None for chunk in chunks)
    assert all(chunk.metadata["section_anchor"] is None for chunk in chunks)


def test_metadata_is_preserved_and_visibility_resolved() -> None:
    doc = make_document(
        "Body text about the club.",
        metadata={"visibility": "members", "extra": 1},
        title="Club Rules",
        source="rules.docx",
        fmt="docx",
    )
    (chunk,) = chunk_document(doc)
    assert chunk.metadata["title"] == "Club Rules"
    assert chunk.metadata["source"] == "rules.docx"
    assert chunk.metadata["format"] == "docx"
    assert chunk.metadata["visibility"] == "members"
    assert chunk.metadata["document_metadata"] == {"visibility": "members", "extra": 1}


def test_visibility_defaults_to_public_and_can_be_overridden() -> None:
    doc = make_document("Body text.")
    assert chunk_document(doc)[0].metadata["visibility"] == "public"
    assert chunk_document(doc, visibility="private")[0].metadata["visibility"] == "private"


def test_chunk_ids_are_unique_and_deterministic() -> None:
    doc = make_document(paragraphs(20))
    first = chunk_document(doc, chunk_size=120, chunk_overlap=20)
    second = chunk_document(doc, chunk_size=120, chunk_overlap=20)
    ids = [chunk.chunk_id for chunk in first]
    assert len(ids) == len(set(ids))
    assert ids == [chunk.chunk_id for chunk in second]
    assert all(chunk.metadata["chunk_id"] == chunk.chunk_id for chunk in first)


def test_returns_chunk_instances() -> None:
    chunks = chunk_document(make_document("Hello world."))
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-10, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_parameters_raise(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(
            make_document("Some text."),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


# --------------------------------------------------------------------------- #
# Tabular documents (spreadsheets): rows are never cut
# --------------------------------------------------------------------------- #


def make_tabular(rows: list[str], *, source: str = "bureau.xlsx") -> ParsedDocument:
    """Build a spreadsheet-shaped document out of rendered row texts."""
    table_rows = [
        TableRow(text=text, container="Bureau", row_number=index + 1, sheet="Bureau")
        for index, text in enumerate(rows)
    ]
    return ParsedDocument(
        text="\n".join(rows),
        source=source,
        format="xlsx",
        title="Bureau",
        tabular=table_rows,
    )


def short_rows(count: int) -> list[str]:
    """Rendered rows well under ``SHEET_MIN_ROW_CHARS``."""
    return [
        f"Fichier : bureau.xlsx | Feuille : Bureau | Nom : Membre{i} | Rôle : Trésorier"
        for i in range(count)
    ]


def test_tabular_short_rows_are_grouped_within_budget() -> None:
    rows = short_rows(10)
    doc = make_tabular(rows)
    chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=50)

    settings = get_settings()
    assert settings.sheet_max_rows_per_chunk == 8
    assert len(chunks) == 2
    assert chunks[0].text.splitlines() == rows[:8]
    assert chunks[1].text.splitlines() == rows[8:]
    assert all(chunk.char_count <= 2000 for chunk in chunks)
    assert "\n".join(chunk.text for chunk in chunks) == doc.text


def test_tabular_never_exceeds_rows_per_chunk() -> None:
    doc = make_tabular(short_rows(20))
    chunks = chunk_document(doc, chunk_size=10_000, chunk_overlap=0)
    settings = get_settings()
    assert len(chunks) == 3  # 8 + 8 + 4
    assert all(
        len(chunk.text.splitlines()) <= settings.sheet_max_rows_per_chunk
        for chunk in chunks
    )


def test_tabular_long_row_is_kept_whole_in_its_own_chunk() -> None:
    long_row = "Fichier : bureau.xlsx | Feuille : Bureau | Commentaire : " + "x" * 600
    rows = [short_rows(1)[0], long_row, short_rows(1)[0].replace("Membre0", "Membre2")]
    doc = make_tabular(rows)
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=20)

    holders = [chunk for chunk in chunks if long_row in chunk.text]
    assert len(holders) == 1
    assert holders[0].text == long_row
    assert holders[0].char_count > 300  # never split, even above chunk_size


def test_tabular_rows_are_never_split_across_chunks() -> None:
    rows = [
        short_rows(1)[0],
        "Fichier : bureau.xlsx | Feuille : Bureau | Note : " + "y" * 400,
        short_rows(1)[0].replace("Membre0", "Membre2"),
    ]
    doc = make_tabular(rows)
    chunks = chunk_document(doc, chunk_size=250, chunk_overlap=10)

    for row in rows:
        holders = [chunk for chunk in chunks if row in chunk.text]
        assert len(holders) == 1
        assert row in holders[0].text.splitlines()
    assert "\n".join(chunk.text for chunk in chunks) == doc.text


def test_tabular_chunks_carry_row_provenance() -> None:
    doc = make_tabular(short_rows(3))
    (chunk, *rest) = chunk_document(doc, chunk_size=10_000, chunk_overlap=0)
    assert rest == []
    assert chunk.metadata["tabular"] is True
    assert chunk.metadata["row_number"] == 1
    assert chunk.metadata["sheet"] == "Bureau"
    assert chunk.metadata["container"] == "Bureau"
    assert chunk.metadata["visibility"] == "public"
    assert chunk.metadata["format"] == "xlsx"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunk_count"] == 1


def test_tabular_documents_without_rows_fall_back_to_text() -> None:
    doc = ParsedDocument(text="not really tabular", source="x.md", format="markdown")
    assert doc.tabular is None
    (chunk,) = chunk_document(doc)
    assert chunk.text == "not really tabular"
