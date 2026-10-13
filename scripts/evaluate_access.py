"""CLI script: 50-question member / non-member access & accuracy evaluation.

Suite: **30 member questions + 20 non-member questions**, with French, English
and Arabic mixed in. Zero LLM calls.

It runs on the labeled public+members corpus defined in ``scripts/evaluate.py``
(the live index only holds public content, so member access cannot be tested
there). Retrieval uses the *shipped* thresholds from configuration: English
questions use ``RETRIEVAL_MIN_SCORE``, French/Arabic use
``RETRIEVAL_MIN_SCORE_MULTILINGUAL``.

A question is **correct** when:
  - it has an expected document and that document is among the kept hits, or
  - it has no expected document (members-only asked by a non-member, or
    out-of-scope) and the retriever refuses (no hit above threshold).

Also reported: Hit@1/5, MRR over answerable questions, a per-language breakdown,
and a member-content leak count for non-member questions (must stay 0).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
for _path in (str(_SRC), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.chdir(_REPO_ROOT)

import evaluate as ev  # noqa: E402  (reuses the labeled corpus + store builder)
from mc_rag.config import get_settings  # noqa: E402
from mc_rag.infrastructure.embeddings.local_embedder import get_embedder  # noqa: E402
from mc_rag.infrastructure.vector_store.chroma_store import (  # noqa: E402
    VISIBILITY_MEMBERS,
    allowed_visibilities,
)
from mc_rag.rag.prompts import question_language  # noqa: E402

MEMBER = "member"
NON_MEMBER = "non_member"
TOP_K = 5

# ---------------------------------------------------------------------------
# 30 member questions (the caller is a member): members-only + public targets,
# mixed languages, plus a few out-of-scope ones that must be refused.
# ---------------------------------------------------------------------------
MEMBER_SET: tuple[ev.Question, ...] = (
    # --- members-only documents (12) ---
    ev.Question(
        "Comment réserver une salle de réunion ?", "fr", MEMBER, ("m_procedures_fr",), True
    ),
    ev.Question("Quelles sont les procédures internes ?", "fr", MEMBER, ("m_procedures_fr",), True),
    ev.Question("Comment se faire rembourser une dépense ?", "fr", MEMBER, ("m_budget_fr",), True),
    ev.Question("Quel budget pour le département IT ?", "fr", MEMBER, ("m_budget_fr",), True),
    ev.Question("Quels sont les contacts du bureau ?", "fr", MEMBER, ("m_contacts_fr",), True),
    ev.Question("À qui m'adresser en urgence ?", "fr", MEMBER, ("m_contacts_fr",), True),
    ev.Question("How do I book the meeting room?", "en", MEMBER, ("m_schedule_en",), True),
    ev.Question("What is the internal schedule?", "en", MEMBER, ("m_schedule_en",), True),
    ev.Question("Who manages the access badges?", "en", MEMBER, ("m_schedule_en",), True),
    ev.Question("كيف أحجز قاعة الاجتماع؟", "ar", MEMBER, ("m_board_ar",), True),
    ev.Question("ما هي معلومات التواصل الداخلية؟", "ar", MEMBER, ("m_board_ar",), True),
    ev.Question("Quand sont effectués les remboursements ?", "fr", MEMBER, ("m_budget_fr",), True),
    # --- public documents (12) ---
    ev.Question("Comment rejoindre le club ?", "fr", MEMBER, ("p_membership_fr",), True),
    ev.Question("Faut-il être étudiant pour adhérer ?", "fr", MEMBER, ("p_membership_fr",), True),
    ev.Question(
        "Quels sont les différents départements ?", "fr", MEMBER, ("p_departments_fr",), True
    ),
    ev.Question("Quelles sont les règles du Discord ?", "fr", MEMBER, ("p_discord_fr",), True),
    ev.Question("Quels événements organise le club ?", "fr", MEMBER, ("p_events_fr",), True),
    ev.Question("How can I join MicroClub?", "en", MEMBER, ("p_membership_en",), True),
    ev.Question("Are there hackathons?", "en", MEMBER, ("p_events_en",), True),
    ev.Question("Tell me about student projects", "en", MEMBER, ("p_projects_en",), True),
    ev.Question("كيف أنضم إلى النادي؟", "ar", MEMBER, ("p_membership_ar",), True),
    ev.Question("ما هي الأنشطة والفعاليات؟", "ar", MEMBER, ("p_activities_ar",), True),
    ev.Question("Est-ce que le spam est interdit ?", "fr", MEMBER, ("p_discord_fr",), True),
    ev.Question("When are the workshops?", "en", MEMBER, ("p_events_en",), True),
    # --- out-of-scope, must be refused (6) ---
    ev.Question("Quel temps fera-t-il demain ?", "fr", MEMBER, (), True),
    ev.Question("Who won the 2022 World Cup?", "en", MEMBER, (), True),
    ev.Question("ما هو سعر البيتكوين اليوم؟", "ar", MEMBER, (), True),
    ev.Question("Quelle est la recette du couscous ?", "fr", MEMBER, (), True),
    ev.Question("What is the best movie tonight?", "en", MEMBER, (), True),
    ev.Question("كيف أطبخ الكسكس؟", "ar", MEMBER, (), True),
)

# ---------------------------------------------------------------------------
# 20 non-member questions: public targets must be answered, members-only
# targets must be refused (no member content visible), plus out-of-scope.
# ---------------------------------------------------------------------------
NON_MEMBER_SET: tuple[ev.Question, ...] = (
    # --- public documents, must be answered (8) ---
    ev.Question("Comment devenir membre ?", "fr", NON_MEMBER, ("p_membership_fr",)),
    ev.Question("Quels pôles techniques existent ?", "fr", NON_MEMBER, ("p_departments_fr",)),
    ev.Question(
        "Quelles sont les règles du serveur Discord ?", "fr", NON_MEMBER, ("p_discord_fr",)
    ),
    ev.Question("What are the membership requirements?", "en", NON_MEMBER, ("p_membership_en",)),
    ev.Question("Are there coding competitions?", "en", NON_MEMBER, ("p_projects_en",)),
    ev.Question("ما هي شروط العضوية في النادي؟", "ar", NON_MEMBER, ("p_membership_ar",)),
    ev.Question("ما هي الفعاليات القادمة؟", "ar", NON_MEMBER, ("p_activities_ar",)),
    ev.Question("Quand a lieu le hackathon ?", "fr", NON_MEMBER, ("p_events_fr",)),
    # --- members-only targets, must be refused (8) ---
    ev.Question("Quel est le budget du département IT ?", "fr", NON_MEMBER),
    ev.Question("Comment réserver la salle de réunion ?", "fr", NON_MEMBER),
    ev.Question("Quels sont les contacts internes du bureau ?", "fr", NON_MEMBER),
    ev.Question("What is the internal schedule?", "en", NON_MEMBER),
    ev.Question("How do I book the meeting room internally?", "en", NON_MEMBER),
    ev.Question("ما هي المعلومات الداخلية للأعضاء؟", "ar", NON_MEMBER),
    ev.Question("كيف أحجز قاعة الاجتماع؟", "ar", NON_MEMBER),
    ev.Question("Comment se faire rembourser une dépense ?", "fr", NON_MEMBER),
    # --- out-of-scope, must be refused (4) ---
    ev.Question("Quelle est la capitale du Japon ?", "fr", NON_MEMBER),
    ev.Question("How do I fix my car engine?", "en", NON_MEMBER),
    ev.Question("ما هو أفضل فندق في دبي؟", "ar", NON_MEMBER),
    ev.Question("What is the price of Bitcoin?", "en", NON_MEMBER),
)


@dataclass(slots=True)
class Outcome:
    """One evaluated question."""

    question: ev.Question
    kept_docs: list[str] = field(default_factory=list)
    best_score: float = 0.0
    rank: int | None = None
    correct: bool = False
    leaked: bool = False
    threshold: float = 0.0

    @property
    def group(self) -> str:
        return self.question.split


def _threshold_for(text: str, settings: object) -> float:
    """Shipped threshold for a question's language."""
    if question_language(text) == "English":
        return settings.retrieval_min_score  # type: ignore[attr-defined]
    return settings.retrieval_min_score_multilingual  # type: ignore[attr-defined]


def evaluate_group(
    store: object,
    embedder: object,
    questions: tuple[ev.Question, ...],
    settings: object,
) -> list[Outcome]:
    """Retrieve for every question and score its correctness."""
    outcomes: list[Outcome] = []
    for question in questions:
        hits = store.search_by_text(  # type: ignore[attr-defined]
            question.text,
            embedder,
            top_k=TOP_K,
            visibilities=allowed_visibilities(is_member=question.is_member),
        )
        threshold = _threshold_for(question.text, settings)
        kept = [hit for hit in hits if hit.score >= threshold]
        kept_docs: list[str] = []
        for hit in kept:
            doc_id = str(hit.metadata.get("doc_id") or "")
            if doc_id and doc_id not in kept_docs:
                kept_docs.append(doc_id)
        expected = set(question.relevant)
        rank = next(
            (position for position, doc in enumerate(kept_docs, start=1) if doc in expected),
            None,
        )
        if expected:
            correct = any(doc in expected for doc in kept_docs)
        else:
            correct = not kept  # must be refused
        outcomes.append(
            Outcome(
                question=question,
                kept_docs=kept_docs,
                best_score=max((hit.score for hit in hits), default=0.0),
                rank=rank,
                correct=correct,
                leaked=any(
                    str(hit.metadata.get("visibility")) == VISIBILITY_MEMBERS for hit in hits
                ),
                threshold=threshold,
            )
        )
    return outcomes


def summarize(title: str, outcomes: list[Outcome]) -> None:
    """Print per-question lines and aggregate metrics for one group."""
    total = len(outcomes)
    correct = sum(1 for outcome in outcomes if outcome.correct)
    answerable = [outcome for outcome in outcomes if outcome.question.relevant]
    refusable = [outcome for outcome in outcomes if not outcome.question.relevant]

    print(title)
    print("=" * len(title))
    for outcome in outcomes:
        status = "OK  " if outcome.correct else "FAIL"
        if outcome.question.relevant:
            expected = ",".join(outcome.question.relevant)
            got = ",".join(outcome.kept_docs[:2]) or "(refuse)"
            target = f"attendu={expected} obtenu={got}"
        else:
            got = ",".join(outcome.kept_docs[:2]) or "(refuse)"
            target = f"attendu=refus obtenu={got}"
        print(
            f"  [{status}] {outcome.question.lang}  {outcome.question.text}\n"
            f"          {target}  best={outcome.best_score:.3f} seuil={outcome.threshold:.2f}"
        )

    print()
    print(f"  Accuracy          : {correct}/{total}  ({correct / total:.1%})")
    if answerable:
        found = [outcome for outcome in answerable if outcome.rank is not None]
        hit1 = sum(1 for outcome in found if outcome.rank == 1)
        mrr = sum(1.0 / outcome.rank for outcome in found if outcome.rank) / len(answerable)
        print(f"  Retrieval Hit@1   : {hit1}/{len(answerable)}  ({hit1 / len(answerable):.1%})")
        print(f"  MRR               : {mrr:.3f}")
    if refusable:
        refused = sum(1 for outcome in refusable if not outcome.kept_docs)
        print(f"  Refus corrects    : {refused}/{len(refusable)}  ({refused / len(refusable):.1%})")
    print()


def language_breakdown(outcomes: list[Outcome]) -> None:
    """Print correctness by question language."""
    print("Par langue")
    print("----------")
    for lang in ("fr", "en", "ar"):
        group = [outcome for outcome in outcomes if outcome.question.lang == lang]
        if not group:
            continue
        correct = sum(1 for outcome in group if outcome.correct)
        print(f"  {lang.upper():3} : {correct}/{len(group)}  ({correct / len(group):.1%})")
    print()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console fix on Windows
        pass

    settings = get_settings()
    embedder = get_embedder()
    store, chunks_total = ev.build_store(ev.CORPUS)

    print()
    print("Evaluation d'acces : 30 membre + 20 non-membre")
    print("-----------------------------------------------")
    print(f"  Corpus             : {len(ev.CORPUS)} documents ({chunks_total} chunks)")
    print(f"  Seuil EN           : {settings.retrieval_min_score:.2f}")
    print(f"  Seuil FR/AR        : {settings.retrieval_min_score_multilingual:.2f}")
    print(f"  top_k              : {TOP_K}")
    print()

    member_outcomes = evaluate_group(store, embedder, MEMBER_SET, settings)
    non_member_outcomes = evaluate_group(store, embedder, NON_MEMBER_SET, settings)
    all_outcomes = member_outcomes + non_member_outcomes

    summarize("Bloc MEMBRE (30 questions)", member_outcomes)
    summarize("Bloc NON-MEMBRE (20 questions)", non_member_outcomes)

    total = len(all_outcomes)
    correct = sum(1 for outcome in all_outcomes if outcome.correct)
    print("Synthese globale")
    print("----------------")
    print(f"  Accuracy totale    : {correct}/{total}  ({correct / total:.1%})")
    leaks = sum(1 for outcome in non_member_outcomes if outcome.leaked)
    print(f"  Fuites membre (non-membre) : {leaks}  {'OK' if leaks == 0 else 'ECHEC'}")
    print()
    language_breakdown(all_outcomes)

    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
