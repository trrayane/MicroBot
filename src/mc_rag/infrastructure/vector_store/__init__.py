"""Package marker for src.mc_rag.infrastructure.vector_store.

Convenience re-exports for the Chroma adapter and visibility helpers.
"""

from mc_rag.infrastructure.vector_store.chroma_store import (
    ALL_VISIBILITIES,
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
    SearchResult,
    allowed_visibilities,
)

__all__ = [
    "ALL_VISIBILITIES",
    "VISIBILITY_MEMBERS",
    "VISIBILITY_PUBLIC",
    "ChromaStore",
    "SearchResult",
    "allowed_visibilities",
]