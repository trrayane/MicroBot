"""Package marker for src.mc_rag.infrastructure.embeddings.

Convenience re-exports for the embedding adapters.
"""

from mc_rag.infrastructure.embeddings.local_embedder import LocalEmbedder, get_embedder

__all__ = [
    "LocalEmbedder",
    "get_embedder",
]