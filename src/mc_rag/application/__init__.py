"""Package marker for src.mc_rag.application.

Convenience re-exports for the application-layer use-case services.
"""

from mc_rag.application.ingestion_service import (
    FileIngestionResult,
    IngestionReport,
    IngestionService,
    IngestionStatus,
    parse_document,
)

__all__ = [
    "FileIngestionResult",
    "IngestionReport",
    "IngestionService",
    "IngestionStatus",
    "parse_document",
]
