"""Query rewriting: the single extra LLM call and its fallbacks.

``QueryRewriter`` must never raise, never retry a 429, and always cap what it
returns. ``QueryPlan.queries`` is the contract the retriever relies on to
decide what gets searched besides the original question.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from mc_rag.domain.exceptions import LLMRateLimitError, LLMUnavailableError
from mc_rag.rag.query_rewriting import QueryPlan, QueryRewriter

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FakeResponse:
    """Minimal stand-in for ``LLMResponse``."""

    text: str


@dataclass(slots=True)
class FakeLLM:
    """LLM double: returns *text* (or raises), recording every call."""

    text: str = ""
    error: Exception | None = None
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def complete(self, messages: Sequence[dict[str, str]]) -> FakeResponse:
        self.calls.append([dict(message) for message in messages])
        if self.error is not None:
            raise self.error
        return FakeResponse(text=self.text)


def rewriter(
    llm: FakeLLM,
    *,
    enabled: bool = True,
    multi_query: bool = False,
    max_chars: int = 300,
    variant_count: int = 2,
    variant_max_chars: int = 200,
) -> QueryRewriter:
    """Build a rewriter with explicit flags (settings stay at their defaults)."""
    return QueryRewriter(
        llm,
        enabled=enabled,
        multi_query=multi_query,
        max_chars=max_chars,
        variant_count=variant_count,
        variant_max_chars=variant_max_chars,
    )


def payload(*, rewritten: str = "hackathon mc club date", variants: Sequence[str] = ()) -> str:
    """A valid JSON answer as the model is instructed to produce."""
    return json.dumps({"rewritten": rewritten, "variants": list(variants)})


# --------------------------------------------------------------------------- #
# Activity / gating
# --------------------------------------------------------------------------- #


def test_rewriting_disabled_by_default() -> None:
    assert QueryRewriter(FakeLLM(), enabled=None, multi_query=None).active is False


def test_active_when_either_feature_is_on() -> None:
    assert rewriter(FakeLLM(), enabled=True, multi_query=False).active is True
    assert rewriter(FakeLLM(), enabled=False, multi_query=True).active is True


def test_inactive_rewriter_spends_no_call() -> None:
    llm = FakeLLM(text=payload())

    plan = rewriter(llm, enabled=False, multi_query=False).plan("question")

    assert llm.calls == []
    assert plan == QueryPlan()


def test_plan_uses_a_single_llm_call_for_rewrite_and_variants() -> None:
    llm = FakeLLM(text=payload(variants=["mc club hackathon", "date hackathon club"]))
    rw = rewriter(llm, enabled=True, multi_query=True)

    plan = rw.plan("c'est quand le hackathon ?")

    assert len(llm.calls) == 1
    assert plan.llm_calls == 1
    assert plan.rewritten == "hackathon mc club date"
    assert plan.variants == ("mc club hackathon", "date hackathon club")


# --------------------------------------------------------------------------- #
# Prompt content
# --------------------------------------------------------------------------- #


def test_prompt_contains_the_question_and_history() -> None:
    llm = FakeLLM(text=payload())

    rewriter(llm).plan(
        "et le deuxième jour ?",
        history=[
            {"role": "user", "content": "Parle du hackathon"},
            {"role": "other", "content": "x"},
        ],
    )

    user = llm.calls[0][-1]["content"]
    assert "et le deuxième jour ?" in user
    assert "Parle du hackathon" in user
    assert "other" not in user  # only user/assistant turns are summarised


def test_prompt_asks_for_variants_only_when_multi_query_is_on() -> None:
    llm = FakeLLM(text=payload())

    rewriter(llm, enabled=True, multi_query=False).plan("question")
    assert "variants\": no" in llm.calls[0][0]["content"]

    llm.calls.clear()
    rewriter(llm, enabled=True, multi_query=True).plan("question")
    assert "variants\": yes (2)" in llm.calls[0][0]["content"]


# --------------------------------------------------------------------------- #
# Parsing and capping
# --------------------------------------------------------------------------- #


def test_json_wrapped_in_code_fences_is_parsed() -> None:
    llm = FakeLLM(text=f"```json\n{payload()}\n```")

    plan = rewriter(llm).plan("question")

    assert plan.rewritten == "hackathon mc club date"


def test_json_with_leading_prose_is_parsed() -> None:
    llm = FakeLLM(text=f"Here you go: {payload()} — hope that helps!")

    plan = rewriter(llm).plan("question")

    assert plan.rewritten == "hackathon mc club date"


def test_unparseable_answer_yields_an_empty_plan_but_costs_one_call() -> None:
    llm = FakeLLM(text="I refuse to answer in JSON.")

    plan = rewriter(llm).plan("question")

    assert plan.rewritten == ""
    assert plan.variants == ()
    assert plan.llm_calls == 1


def test_empty_answer_yields_an_empty_plan() -> None:
    llm = FakeLLM(text="")

    plan = rewriter(llm).plan("question")

    assert len(llm.calls) == 1
    assert plan.rewritten == ""
    assert plan.variants == ()


def test_rate_limit_is_never_retried() -> None:
    llm = FakeLLM(error=LLMRateLimitError("429 slow down"))

    plan = rewriter(llm).plan("question")

    assert len(llm.calls) == 1  # exactly one attempt, no retry
    assert plan.llm_calls == 1
    assert plan.rewritten == ""
    assert plan.variants == ()


def test_unavailable_llm_degrades_to_an_empty_plan() -> None:
    llm = FakeLLM(error=LLMUnavailableError("down"))

    plan = rewriter(llm).plan("question")

    assert len(llm.calls) == 1
    assert plan == QueryPlan(llm_calls=1)


def test_rewritten_query_is_capped_at_max_chars() -> None:
    long_query = "mot " * 200
    llm = FakeLLM(text=payload(rewritten=long_query))

    plan = rewriter(llm, max_chars=50).plan("question")

    assert len(plan.rewritten) <= 50


def test_variants_are_capped_in_count_and_length() -> None:
    many = [f"variante numero {i} " + "x" * 300 for i in range(6)]
    llm = FakeLLM(text=payload(rewritten="q1", variants=many))

    plan = rewriter(llm, multi_query=True, variant_count=2, variant_max_chars=40).plan("question")

    assert len(plan.variants) == 2
    assert all(len(variant) <= 40 for variant in plan.variants)


def test_variant_that_only_repeats_the_question_is_dropped() -> None:
    question = "Quand est le hackathon du MC Club ?"
    llm = FakeLLM(
        text=payload(
            rewritten="hackathon mc club date",
            variants=[question.lower(), "mc club date"],
        )
    )

    plan = rewriter(llm, multi_query=True).plan(question)

    # Fully contained in the question's own words: no extra recall.
    assert "quand est le hackathon du mc club ?" not in plan.variants
    assert plan.variants == ("mc club date",)


def test_duplicate_variants_are_deduped_case_insensitively() -> None:
    llm = FakeLLM(text=payload(variants=["Date Hackathon", "date hackathon", "hackathon date"]))

    plan = rewriter(llm, multi_query=True, variant_count=3).plan("question")

    assert plan.variants == ("Date Hackathon", "hackathon date")


def test_variants_are_disabled_when_multi_query_is_off() -> None:
    llm = FakeLLM(text=payload(variants=["ignored variant"]))

    plan = rewriter(llm, enabled=True, multi_query=False).plan("question")

    assert plan.variants == ()
    assert plan.queries == ["hackathon mc club date"]


def test_rewritten_equal_to_the_question_is_discarded() -> None:
    llm = FakeLLM(text=payload(rewritten="  QUAND EST LE HACKATHON ?  "))

    plan = rewriter(llm).plan("Quand est le hackathon ?")

    assert plan.rewritten == ""


def test_non_string_payload_fields_are_ignored() -> None:
    llm = FakeLLM(text='{"rewritten": 42, "variants": ["ok", 7]}')

    plan = rewriter(llm, multi_query=True).plan("question")

    assert plan.rewritten == ""
    assert plan.variants == ("ok",)


# --------------------------------------------------------------------------- #
# QueryPlan.queries
# --------------------------------------------------------------------------- #


def test_queries_dedupes_and_keeps_order() -> None:
    plan = QueryPlan(rewritten="B", variants=("b", "C", "C", ""))

    assert plan.queries == ["B", "C"]


def test_empty_plan_queries_is_empty() -> None:
    assert QueryPlan().queries == []


def test_plan_with_only_variants_skips_the_empty_rewrite() -> None:
    plan = QueryPlan(rewritten="", variants=("variant",))

    assert plan.queries == ["variant"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"rewritten": "a", "variants": ("a",)}, ["a"]),
        ({"rewritten": "a b", "variants": ("a b", "c")}, ["a b", "c"]),
    ],
)
def test_queries_collapses_duplicates(kwargs: dict[str, Any], expected: list[str]) -> None:
    plan = QueryPlan(**kwargs)  # type: ignore[arg-type]

    assert plan.queries == expected
