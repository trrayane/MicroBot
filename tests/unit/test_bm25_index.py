"""Unit tests for the SQLite FTS5 (BM25) lexical index."""

from __future__ import annotations

from pathlib import Path

import pytest

from mc_rag.domain.exceptions import LexicalIndexError
from mc_rag.infrastructure.lexical_store.bm25_index import (
    BM25Index,
    IndexEntry,
    LexicalHit,
)

MEMBERS = "members"
PUBLIC = "public"


def entry(
    chunk_id: str,
    text: str,
    *,
    visibility: str = PUBLIC,
    doc_id: str = "doc-1",
    **extra: object,
) -> IndexEntry:
    """Build an :class:`IndexEntry` with the metadata shape of the chunker."""
    metadata: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": f"{doc_id}.md",
        "visibility": visibility,
        "title": "Règles du club",
        **extra,
    }
    return IndexEntry(chunk_id=chunk_id, text=text, metadata=metadata)


@pytest.fixture
def index(tmp_path: Path) -> BM25Index:
    """A file-backed index so persistence can be asserted too."""
    return BM25Index(tmp_path / "bm25.sqlite3")


def test_upsert_makes_chunks_searchable(index: BM25Index) -> None:
    index.upsert(
        [
            entry("c1", "Le trésorier valide le budget du semestre."),
            entry("c2", "Les inscriptions au hackathon ouvrent en mars."),
        ]
    )

    hits = index.search("budget trésorier", visibilities=(PUBLIC,), limit=10)

    assert [hit.chunk_id for hit in hits] == ["c1"]
    assert hits[0].entry.text.startswith("Le trésorier")
    assert hits[0].entry.metadata["title"] == "Règles du club"


def test_search_is_case_and_accent_insensitive(index: BM25Index) -> None:
    index.upsert([entry("c1", "Le Rôle du trésorier est précisé ici.")])

    assert [h.chunk_id for h in index.search("ROLE", visibilities=(PUBLIC,))] == ["c1"]
    assert [h.chunk_id for h in index.search("trésorier", visibilities=(PUBLIC,))] == ["c1"]


def test_search_matches_substrings_of_words_only_as_whole_tokens(index: BM25Index) -> None:
    index.upsert([entry("c1", "budget"), entry("c2", "budgétaire")])

    hits = index.search("budget", visibilities=(PUBLIC,))
    assert [hit.chunk_id for hit in hits] == ["c1"]


def test_search_orders_by_bm25_best_first(index: BM25Index) -> None:
    index.upsert(
        [
            entry("rare", "le budget du hackathon et du bureau et du semester"),
            entry("common", "budget"),
        ]
    )

    hits = index.search("budget", visibilities=(PUBLIC,), limit=10)

    assert {hit.chunk_id for hit in hits} == {"rare", "common"}
    assert hits[0].bm25_score <= hits[-1].bm25_score  # smaller is better
    assert all(hit.force >= 0 for hit in hits)


def test_search_without_token_returns_no_hit(index: BM25Index) -> None:
    index.upsert([entry("c1", "du texte ici")])

    assert index.search("   ??? --- ", visibilities=(PUBLIC,)) == []
    assert index.search("", visibilities=(PUBLIC,)) == []


def test_digits_are_indexed_and_searchable(index: BM25Index) -> None:
    index.upsert([entry("c1", "réunion du 12/03 à 18h")])

    assert [h.chunk_id for h in index.search("12 03", visibilities=(PUBLIC,))] == ["c1"]
    assert index.search("motintrouvable", visibilities=(PUBLIC,)) == []


def test_search_respects_the_limit(index: BM25Index) -> None:
    index.upsert([entry(f"c{i}", f"budget numero {i}") for i in range(5)])

    assert len(index.search("budget", visibilities=(PUBLIC,), limit=2)) == 2
    with pytest.raises(LexicalIndexError, match="positive"):
        index.search("budget", visibilities=(PUBLIC,), limit=0)


def test_upsert_is_idempotent(index: BM25Index) -> None:
    index.upsert([entry("c1", "première version du texte")])
    index.upsert([entry("c1", "première version du texte")])

    assert index.count() == 1
    assert len(index.search("première", visibilities=(PUBLIC,))) == 1


def test_upsert_replaces_changed_text(index: BM25Index) -> None:
    index.upsert([entry("c1", "ancien contenu sur les inscriptions")])
    index.upsert([entry("c1", "nouveau contenu sur les réunions")])

    assert index.count() == 1
    assert index.search("inscriptions", visibilities=(PUBLIC,)) == []
    assert [h.chunk_id for h in index.search("réunions", visibilities=(PUBLIC,))] == ["c1"]


def test_delete_document_only_removes_its_chunks(index: BM25Index) -> None:
    index.upsert(
        [
            entry("a", "budget du document A", doc_id="doc-a"),
            entry("b", "budget du document B", doc_id="doc-b"),
        ]
    )

    assert index.delete_document("doc-a") == 1
    assert index.count() == 1
    assert [h.chunk_id for h in index.search("budget", visibilities=(PUBLIC,))] == ["b"]
    assert index.delete_document("doc-a") == 0  # already gone


def test_clear_empties_both_tables(index: BM25Index) -> None:
    index.upsert([entry("a", "budget"), entry("b", "budget")])

    assert index.clear() == 2
    assert index.count() == 0
    assert index.search("budget", visibilities=(PUBLIC,)) == []


def test_unknown_visibility_is_rejected(index: BM25Index) -> None:
    index.upsert([entry("a", "budget")])

    with pytest.raises(LexicalIndexError, match="Unknown visibility"):
        index.search("budget", visibilities=("secret",))
    with pytest.raises(LexicalIndexError, match="visibility must be allowed"):
        index.search("budget", visibilities=())


def test_derived_doc_id_is_stable_without_metadata(index: BM25Index) -> None:
    bare = IndexEntry(chunk_id="c1", text="budget", metadata={"source": "a.md"})
    same = IndexEntry(chunk_id="c2", text="budget", metadata={"source": "a.md"})

    assert bare.doc_id == same.doc_id
    assert bare.doc_id != "unknown-doc"
    assert IndexEntry(chunk_id="c3", text="x", metadata={}).doc_id == "unknown-doc"


def test_index_is_persisted_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "bm25.sqlite3"
    writer = BM25Index(path)
    writer.upsert([entry("c1", "budget persistant")])
    writer.close()

    reader = BM25Index(path)
    assert path.is_file()
    assert [h.chunk_id for h in reader.search("persistant", visibilities=(PUBLIC,))] == ["c1"]
    reader.close()


def test_search_on_an_empty_index_is_harmless(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "fresh.sqlite3")

    assert index.search("n'importe quoi", visibilities=(PUBLIC,)) == []
    assert index.count() == 0


def test_hit_force_is_the_negated_bm25_score() -> None:
    hit = LexicalHit(entry=entry("c1", "budget"), bm25_score=-3.5)

    assert hit.force == 3.5
    assert hit.chunk_id == "c1"
