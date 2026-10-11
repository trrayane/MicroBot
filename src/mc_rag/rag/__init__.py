"""Package marker for src.mc_rag.rag.

Convenience re-exports for the RAG building blocks.
"""

from mc_rag.rag.chunking import Chunk, chunk_document

__all__ = [
    "Chunk",
    "chunk_document",
]
