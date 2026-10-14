"""Text normalization shared by the lexical index and its queries.

Both sides of the FTS5 ``MATCH`` expression must be tokenized identically, and
the FTS5 ``unicode61`` tokenizer is deliberately simple. So the index stores a
**pre-normalized** copy of every chunk (built by :func:`normalize`) and the
retriever builds its ``MATCH`` expression from the very same function
(:func:`build_match_expression`).

Normalization is language neutral: case folding, removal of combining marks
(so ``rôle`` == ``role`` and ``ÉTÉ`` == ``été``) and extraction of word runs,
which keeps digits and Arabic letters while dropping punctuation, URLs syntax
and underscores.
"""

from __future__ import annotations

import re
import unicodedata

# Upper bound on the number of terms sent to FTS5 (keeps queries cheap).
MAX_QUERY_TOKENS = 32

# Word characters, underscore excluded: ``rôle`` -> one token, ``col_name`` -> two.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Return *text* case-folded, accent-stripped and word-split by single spaces."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(_WORD_RE.findall(stripped))


def tokenize(text: str, *, max_tokens: int = MAX_QUERY_TOKENS) -> list[str]:
    """Split *text* into normalized, de-duplicated tokens (order preserved)."""
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for token in normalize(text).split():
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def build_match_expression(query: str, *, max_tokens: int = MAX_QUERY_TOKENS) -> str | None:
    """Build the FTS5 ``MATCH`` expression for *query*, or ``None`` if empty.

    Tokens are quoted (they only contain word characters, so no FTS5 operator
    can be injected) and combined with ``OR``: the lexical index is a candidate
    generator for the hybrid ranker, so recall matters more than precision, and
    BM25 weighting already penalises common terms.

    Returns:
        e.g. ``'"tresorier" OR "budget"'``, or ``None`` when *query* holds no token.
    """
    tokens = tokenize(query, max_tokens=max_tokens)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)
