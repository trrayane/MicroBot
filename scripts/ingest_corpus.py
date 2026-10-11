"""CLI script: ingest the raw corpus into the index.

Scans a directory recursively, parses each supported file, chunks and embeds
it, then writes the result into the Chroma collection. Re-running is cheap:
unchanged files are skipped via their stored content hash unless ``--force`` is
given.

Usage::

    python scripts/ingest_corpus.py [PATH] [options]

Run ``python scripts/ingest_corpus.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mc_rag.application.ingestion_service import (
    IngestionReport,
    IngestionService,
)
from mc_rag.config import get_settings
from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    ChromaStore,
)

logger = logging.getLogger("ingest_corpus")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="ingest_corpus",
        description="Parse, chunk, embed and index the raw corpus into Chroma.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=settings.raw_data_dir,
        help="Directory to scan recursively (default: %(default)s).",
    )
    parser.add_argument(
        "--collection",
        default=settings.chroma_collection,
        help="Chroma collection name (default: %(default)s).",
    )
    parser.add_argument(
        "--persist-dir",
        default=settings.chroma_persist_dir,
        help="Directory for the persistent index (default: %(default)s).",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Build an in-memory index instead of persisting to disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and re-index files even when they are unchanged.",
    )
    parser.add_argument(
        "--visibility",
        choices=list(ALL_VISIBILITIES),
        default=None,
        help="Override the folder-inferred visibility for every file.",
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
    service = IngestionService(store=store)

    try:
        report = service.ingest_directory(
            Path(args.path),
            visibility=args.visibility,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
        logger.error("Ingestion aborted: %s", exc)
        return 2

    print_summary(report)
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
    print()


if __name__ == "__main__":
    raise SystemExit(main())
