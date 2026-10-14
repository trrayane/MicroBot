"""Conditional query rewriting and multi-query expansion (at most **one** LLM call).

When the first retrieval pass looks weak (or the question is a follow-up), a
single LLM call rewrites the question into a standalone, keyword-rich query and
may add a couple of alternative phrasings. Both features share that call:

* ``QUERY_REWRITE_ENABLED`` -> use the rewritten query;
* ``MULTI_QUERY_ENABLED``   -> also search the variants, fused with the others.

Budget rules, enforced by :class:`QueryRewriter` and the retriever:

* **zero** LLM call when retrieval already found candidates above the score
  threshold and there is no conversation history;
* **zero** LLM call when the first pass found *nothing at all* (the pipeline
  refuses without spending anything);
* **one** call otherwise, producing the rewritten query *and* the variants
  together - the answer call stays the only other one;
* a rate-limited call (429) is never retried: the plan degrades to the
  heuristic fallback instead.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import LLMError

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


class _LLM(Protocol):
    """Minimal surface of the chat client used for rewriting."""

    def complete(self, messages: Sequence[dict[str, str]]) -> Any: ...


@dataclass(slots=True, frozen=True)
class QueryPlan:
    """Queries derived from one question by a single rewriting call."""

    rewritten: str = ""
    variants: tuple[str, ...] = ()
    llm_calls: int = 0

    @property
    def queries(self) -> list[str]:
        """The extra queries to search, in order and without duplicates."""
        ordered: list[str] = [self.rewritten, *self.variants]
        seen: set[str] = set()
        result: list[str] = []
        for query in ordered:
            cleaned = _clean(query)
            if not cleaned or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            result.append(cleaned)
        return result


class QueryRewriter:
    """Turns a weak question into a rewritten query plus optional variants."""

    def __init__(
        self,
        llm: _LLM,
        *,
        enabled: bool | None = None,
        multi_query: bool | None = None,
        max_chars: int | None = None,
        variant_count: int | None = None,
        variant_max_chars: int | None = None,
    ) -> None:
        settings = get_settings()
        self._llm = llm
        self.enabled = settings.query_rewrite_enabled if enabled is None else enabled
        self.multi_query = settings.multi_query_enabled if multi_query is None else multi_query
        self.max_chars = max_chars if max_chars is not None else settings.query_rewrite_max_chars
        self.variant_count = (
            variant_count if variant_count is not None else settings.multi_query_count
        )
        self.variant_max_chars = (
            variant_max_chars
            if variant_max_chars is not None
            else settings.multi_query_max_chars
        )

    @property
    def active(self) -> bool:
        """Whether at least one rewriting feature is switched on."""
        return self.enabled or self.multi_query

    def plan(
        self,
        message: str,
        *,
        history: Sequence[dict[str, str]] | None = None,
    ) -> QueryPlan:
        """Return the plan for *message*, spending at most one LLM call.

        Returns:
            An empty plan when the call fails, is rate limited or returns
            something unparseable: the caller keeps the original query.
        """
        if not self.active:
            return QueryPlan()

        messages = self._build_messages(message, history)
        try:
            response = self._llm.complete(messages)
        except LLMError as exc:
            # 429 is an LLMRateLimitError: never retried, we degrade silently.
            logger.warning("Query rewriting skipped (%s): %s", type(exc).__name__, exc)
            return QueryPlan(llm_calls=1)
        except Exception as exc:  # noqa: BLE001 - rewriting is never fatal
            logger.warning("Query rewriting failed: %s", exc)
            return QueryPlan(llm_calls=1)

        rewritten, variants = _parse_payload(getattr(response, "text", "") or "")
        plan = QueryPlan(
            rewritten=self._rewrite(rewritten, message),
            variants=self._variants(variants, message),
            llm_calls=1,
        )
        logger.debug(
            "Query rewriting produced rewritten=%r variants=%d",
            plan.rewritten,
            len(plan.variants),
        )
        return plan

    def _build_messages(
        self, message: str, history: Sequence[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        """Assemble the single rewriting prompt (system + user)."""
        want_rewrite = "yes" if self.enabled else "no"
        want_variants = f"yes ({self.variant_count})" if self.multi_query else "no"
        rules = [
            "Rewrite the user question so a keyword search over a small club",
            "corpus can retrieve the relevant documents.",
            "Return ONLY compact JSON: {\"rewritten\": \"...\", \"variants\": [\"...\"]}.",
            f"Include \"rewritten\": {want_rewrite}.",
            f"Include \"variants\": {want_variants}.",
            f"- \"rewritten\": at most {self.max_chars} characters, standalone,",
            "keywords first, same language as the question.",
            f"- each variant: at most {self.variant_max_chars} characters, a",
            "different phrasing of the same intent.",
            "- Never answer the question. Never invent facts.",
            "- No markdown, no explanation, JSON only.",
        ]
        context = _history_summary(history)
        user_parts = [f"Question: {message.strip()}"]
        if context:
            user_parts.append(f"Conversation so far: {context}")
        return [
            {"role": "system", "content": " ".join(rules)},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    def _rewrite(self, candidate: str, message: str) -> str:
        """Sanitize the rewritten query (empty when it adds nothing)."""
        cleaned = _clean(candidate)[: self.max_chars].strip()
        if not cleaned:
            return ""
        if cleaned.lower() == _clean(message).lower():
            return ""
        return cleaned

    def _variants(self, candidates: Sequence[str], message: str) -> tuple[str, ...]:
        """Sanitize the variants: count, length and duplicates capped."""
        if not self.multi_query or self.variant_count <= 0:
            return ()
        originals = {part.lower() for part in _clean(message).lower().split()}
        kept: list[str] = []
        for candidate in candidates:
            cleaned = _clean(candidate)[: self.variant_max_chars].strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in {v.lower() for v in kept}:
                continue
            if lowered == _clean(message).lower():
                continue
            # A variant made only of the question's own words adds no recall.
            if originals and set(lowered.split()) <= originals:
                continue
            kept.append(cleaned)
            if len(kept) >= self.variant_count:
                break
        return tuple(kept)


def _clean(text: str) -> str:
    """Collapse whitespace and strip the ends of *text*."""
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def _history_summary(history: Sequence[dict[str, str]] | None) -> str:
    """Short, role-tagged summary of the last turns (empty when no history)."""
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-4:]:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {_clean(content)[:200]}")
    return " | ".join(lines)


def _parse_payload(raw: str) -> tuple[str, list[str]]:
    """Extract ``(rewritten, variants)`` from an LLM answer.

    Tolerates code fences, leading prose and trailing punctuation; anything
    unparseable yields an empty plan (no exception escapes).
    """
    text = (raw or "").strip()
    if not text:
        return "", []

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return "", []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.debug("Unparseable rewriting payload: %.120s", raw)
        return "", []
    if not isinstance(payload, dict):
        return "", []

    rewritten = payload.get("rewritten")
    raw_variants = payload.get("variants")
    variants: list[str] = []
    if isinstance(raw_variants, list):
        variants = [str(item) for item in raw_variants if isinstance(item, str)]
    return (str(rewritten) if isinstance(rewritten, str) else ""), variants
