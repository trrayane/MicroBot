"""Package marker for the lexical (BM25 / SQLite FTS5) index."""

from mc_rag.infrastructure.lexical_store.bm25_index import (
    BM25Index,
    IndexEntry,
    LexicalHit,
)
from mc_rag.infrastructure.lexical_store.tokenizer import (
    build_match_expression,
    normalize,
    tokenize,
)

__all__ = [
    "BM25Index",
    "IndexEntry",
    "LexicalHit",
    "build_match_expression",
    "normalize",
    "tokenize",
]
