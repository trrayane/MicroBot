"""CLI script: rebuild the search indexes.

Two modes:

* **BM25 (default)** - every chunk stored in Chroma is read back and written
  into the SQLite FTS5 index. No embedding happens, so this is the cheap way to
  give an existing corpus lexical search (needed by ``HYBRID_ENABLED=true``)
  and to repair the BM25 index after a crash or a schema change.
* **``--reingest``** - the corpus is re-parsed, re-embedded and re-indexed in
  Chroma; the ingestion service refreshes the BM25 index as it goes (stale
  entries are cleared first).

Usage::

    python scripts/rebuild_index.py                  # rebuild BM25 from Chroma
    python scripts/rebuild_index.py --reingest       # full re-ingest (+ BM25)
    python scripts/rebuild_index.py --reingest --force

Run ``python scripts/rebuild_index.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mc_rag.application.ingestion_service import IngestionReport, IngestionService
from mc_rag.config import get_settings
from mc_rag.domain.exceptions import DomainError
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index, IndexEntry
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore

logger = logging.getLogger("rebuild_index")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="rebuild_index",
        description="Rebuild the BM25 index from Chroma, or re-ingest the corpus.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-parse and re-embed the corpus into Chroma (and refresh BM25).",
    )
    parser.add_argument(
        "--path",
        default=settings.raw_data_dir,
        help="Corpus directory used with --reingest (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --reingest, re-embed files even when unchanged.",
    )
    parser.add_argument(
        "--collection",
        default=settings.chroma_collection,
        help="Chroma collection name (default: %(default)s).",
    )
    parser.add_argument(
        "--persist-dir",
        default=settings.chroma_persist_dir,
        help="Directory of the persistent Chroma index (default: %(default)s).",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Read an in-memory Chroma collection instead of the on-disk one.",
    )
    parser.add_argument(
        "--bm25-path",
        default=settings.bm25_index_path,
        help="SQLite file of the BM25 index (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size when reading chunks from Chroma (default: %(default)s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 failures, 2 aborted)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    store = ChromaStore(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        persist=not args.ephemeral,
    )
    index = BM25Index(args.bm25_path)

    if args.reingest:
        return _reingest(args, store, index)
    return _rebuild_bm25(store, index, batch_size=args.batch_size)


def _rebuild_bm25(store: ChromaStore, index: BM25Index, *, batch_size: int) -> int:
    """Clear the lexical index and refill it from the vector store."""
    try:
        chunks = store.get_all_chunks(batch_size=batch_size)
        entries = [
            IndexEntry(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]
        cleared = index.clear()
        written = index.upsert(entries)
    except (DomainError, OSError) as exc:
        logger.error("BM25 rebuild aborted: %s", exc)
        return 2

    print()
    print("BM25 index rebuild")
    print("------------------")
    print(f"  Source chunks    : {len(chunks)}")
    print(f"  Stale removed    : {cleared}")
    print(f"  Index file       : {index.path}")
    print(f"  Chunks indexed   : {written}")
    print(f"  Chunks in index  : {index.count()}")
    print()
    if not chunks:
        logger.warning("The vector store is empty: nothing to index.")
    return 0


def _reingest(args: argparse.Namespace, store: ChromaStore, index: BM25Index) -> int:
    """Re-ingest the corpus, refreshing both indexes."""
    service = IngestionService(store=store, lexical=index)
    try:
        cleared = index.clear()
        logger.info("Cleared %d stale lexical entries", cleared)
        report = service.ingest_directory(Path(args.path), force=args.force)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
        logger.error("Re-ingestion aborted: %s", exc)
        return 2

    print_summary(report)
    print(f"  Lexical chunks   : {index.count()}")
    print()
    return 1 if report.failed else 0


def print_summary(report: IngestionReport) -> None:
    """Print the human-readable ingestion summary."""
    print()
    print("Ingestion summary")
    print("-----------------")
    print(f"  Root scanned    : {report.root}")
    print(f"  Files processed : {report.total_files}")
    print(f"  Indexed         : {report.indexed}")
    print(f"  Updated         : {report.updated}")
    print(f"  Unchanged       : {report.unchanged}")
    print(f"  Failed          : {report.failed}")
    print(f"  Chunks embedded : {report.chunks_embedded}")
    print(f"  Time taken      : {report.elapsed_seconds:.2f}s")

    if report.errors:
        print()
        print("Errors")
        print("------")
        for result in report.errors:
            print(f"  - {result.path}: {result.error}")


if __name__ == "__main__":
    raise SystemExit(main())
