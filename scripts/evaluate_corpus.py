"""CLI script: retrieval-only evaluation against the LIVE indexed corpus.

Unlike ``scripts/evaluate.py`` (which builds an isolated in-memory corpus), this
harness queries the **persistent Chroma index** that the service actually uses,
so it reports how the bot retrieves from the documents you ingested for real.

It makes **zero LLM calls**. Each question is labeled with one or more expected
keywords that must appear in a relevant chunk (the corpus is a single document,
so chunk-level matching is what matters here). Cross-lingual questions (French /
Arabic) against the English-only source are included on purpose.

Usage
-----
::

    python scripts/evaluate_corpus.py
    python scripts/evaluate_corpus.py --min-score 0.80 --candidates 10
    python scripts/evaluate_corpus.py --show-top 3
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Relative paths ("./storage/chroma", ".env", ...) resolve against the project
# root, so the script behaves the same wherever it is launched from.
os.chdir(_REPO_ROOT)

from mc_rag.config import get_settings  # noqa: E402
from mc_rag.infrastructure.embeddings.local_embedder import get_embedder  # noqa: E402
from mc_rag.infrastructure.vector_store.chroma_store import (  # noqa: E402
    VISIBILITY_MEMBERS,
    ChromaStore,
    allowed_visibilities,
)

K_VALUES = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class LiveQuestion:
    """A question with the keywords that must appear in a relevant chunk."""

    text: str
    lang: str
    keywords: tuple[str, ...] = ()


# In-scope questions: keywords come from the actual Code of Conduct content.
IN_SCOPE: tuple[LiveQuestion, ...] = (
    LiveQuestion("What happens if a member violates the code of conduct?", "en", ("expulsion",)),
    LiveQuestion("In what language must club communications be conducted?", "en", ("English",)),
    LiveQuestion("What is the club's policy on harassment?", "en", ("zero-tolerance",)),
    LiveQuestion("Are members allowed to discuss politics or religion?", "en", ("politics",)),
    LiveQuestion("What are the sanctions for misconduct?", "en", ("Formal Warning",)),
    LiveQuestion("Who can enter the Headquarters?", "en", ("members-only",)),
    LiveQuestion("What happens to personal items left unattended?", "en", ("20 minutes",)),
    LiveQuestion(
        "Can I take club equipment out of the HQ?", "en", ("removed from the Headquarters",)
    ),
    LiveQuestion("How is the Executive Board elected?", "en", ("secret ballot",)),
    LiveQuestion("Are interviews part of the registration process?", "en", ("Interviews",)),
    LiveQuestion(
        "Is presence on Discord required for active membership?", "en", ("Discord server",)
    ),
    LiveQuestion("What are Class A behavioral offenses?", "en", ("Class A",)),
    LiveQuestion("How can a member resolve a conflict or dispute?", "en", ("Mediation",)),
    LiveQuestion("Are disciplinary matters kept confidential?", "en", ("confidentially",)),
    LiveQuestion("What privileges do alumni have?", "en", ("Alumni",)),
    LiveQuestion("How does the recognition system reward members?", "en", ("Honorable Mention",)),
    # --- cross-lingual: French question, English source ---
    LiveQuestion("Dans quelle langue le club communique-t-il ?", "fr", ("English",)),
    LiveQuestion("Puis-je parler de politique ou de religion au club ?", "fr", ("politics",)),
    LiveQuestion("Quelles sanctions en cas de mauvaise conduite ?", "fr", ("Formal Warning",)),
    LiveQuestion("Que deviennent les objets laisses sans surveillance ?", "fr", ("20 minutes",)),
    LiveQuestion("Le quartier general est-il reserve aux membres ?", "fr", ("members-only",)),
    # --- cross-lingual: Arabic question, English source ---
    LiveQuestion("ما هي اللغة المعتمدة في تواصل النادي؟", "ar", ("English",)),
    LiveQuestion("هل يمكن مناقشة السياسة والدين في النادي؟", "ar", ("politics",)),
    LiveQuestion("ما هي العقوبات في حالة سوء السلوك؟", "ar", ("Formal Warning",)),
)

# Out-of-scope questions: no keyword, must fall below the threshold (be refused).
OUT_OF_SCOPE: tuple[LiveQuestion, ...] = (
    LiveQuestion("What is the weather in Algiers today?", "en"),
    LiveQuestion("Who won the 2022 FIFA World Cup?", "en"),
    LiveQuestion("Give me a recipe for couscous.", "en"),
    LiveQuestion("What is the current price of Bitcoin?", "en"),
    LiveQuestion("How do I fix my car engine?", "en"),
    LiveQuestion("Quelle est la capitale du Japon ?", "fr"),
    LiveQuestion("Donne-moi un film a regarder ce soir.", "fr"),
    LiveQuestion("ما هو سعر البيتكوين اليوم؟", "ar"),
    LiveQuestion("كيف أطبخ الكسكس؟", "ar"),
)


@dataclass(slots=True)
class Outcome:
    """Retrieval outcome for one question."""

    question: LiveQuestion
    score: float
    rank: int | None
    top_index: int | None
    top_score: float
    top_snippet: str
    leaked: bool


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    haystack = text.casefold()
    return any(keyword.casefold() in haystack for keyword in keywords)


def evaluate(store: ChromaStore, candidates: int) -> tuple[list[Outcome], list[Outcome]]:
    """Run in-scope and out-of-scope questions against the live index."""
    embedder = get_embedder()
    visibilities = allowed_visibilities(is_member=False)

    in_scope: list[Outcome] = []
    for question in IN_SCOPE:
        hits = store.search_by_text(
            question.text, embedder, top_k=candidates, visibilities=visibilities
        )
        best = max((hit.score for hit in hits), default=0.0)
        rank: int | None = None
        for position, hit in enumerate(hits, start=1):
            if _contains(hit.text, question.keywords):
                rank = position
                break
        top = hits[0] if hits else None
        in_scope.append(
            Outcome(
                question=question,
                score=best,
                rank=rank,
                top_index=(top.metadata.get("chunk_index") if top else None),
                top_score=(top.score if top else 0.0),
                top_snippet=(top.text[:90].replace("\n", " ") if top else ""),
                leaked=any(
                    str(hit.metadata.get("visibility")) == VISIBILITY_MEMBERS for hit in hits
                ),
            )
        )

    out_of_scope: list[Outcome] = []
    for question in OUT_OF_SCOPE:
        hits = store.search_by_text(
            question.text, embedder, top_k=candidates, visibilities=visibilities
        )
        best = max((hit.score for hit in hits), default=0.0)
        top = hits[0] if hits else None
        out_of_scope.append(
            Outcome(
                question=question,
                score=best,
                rank=None,
                top_index=(top.metadata.get("chunk_index") if top else None),
                top_score=(top.score if top else 0.0),
                top_snippet=(top.text[:90].replace("\n", " ") if top else ""),
                leaked=any(
                    str(hit.metadata.get("visibility")) == VISIBILITY_MEMBERS for hit in hits
                ),
            )
        )
    return in_scope, out_of_scope


def ranking_report(outcomes: list[Outcome]) -> None:
    total = len(outcomes)
    found = [outcome for outcome in outcomes if outcome.rank is not None]
    hits = {k: sum(1 for o in found if o.rank is not None and o.rank <= k) for k in K_VALUES}
    mrr = sum(1.0 / o.rank for o in found if o.rank is not None) / total if total else 0.0
    print("Classement (chunk pertinent trouve)")
    print("-----------------------------------")
    for k in K_VALUES:
        print(f"  Hit@{k}           : {hits[k] / total:.3f}  ({hits[k]}/{total})")
    print(f"  MRR              : {mrr:.3f}")
    print()


def refusal_report(outcomes: list[Outcome], threshold: float) -> None:
    accepted = sum(1 for o in outcomes if o.score >= threshold)
    print("Refus (hors-sujet)")
    print("------------------")
    print(f"  Seuil            : {threshold:.2f}")
    print(f"  Refuses          : {len(outcomes) - accepted}/{len(outcomes)}")
    for outcome in outcomes:
        status = "ACCEPTE (tort)" if outcome.score >= threshold else "refuse (ok)"
        print(f"    [{status}] score={outcome.score:.3f}  {outcome.question.text!r}")
    print()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console fix on Windows
        pass

    parser = argparse.ArgumentParser(
        prog="evaluate_corpus",
        description="Retrieval-only evaluation against the live indexed corpus.",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--show-top", type=int, default=1, help="Snippets shown per question.")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = ChromaStore()
    threshold = settings.retrieval_min_score if args.min_score is None else args.min_score

    documents = store.list_documents()
    print()
    print("Index interroge")
    print("---------------")
    print(f"  Collection         : {store.collection_name}")
    print(f"  Chunks             : {store.count()}")
    print(f"  Documents          : {len(documents)}")
    for row in documents:
        print(f"    - {row['title']} [{row['visibility']}] {row['chunks']} chunks")
    print(f"  Modele embedding   : {settings.embedding_model}")
    print(f"  Seuil retrieval    : {threshold:.2f}")
    print()

    in_scope, out_of_scope = evaluate(store, args.candidates)

    print("Resultats par question (in-scope)")
    print("---------------------------------")
    for outcome in in_scope:
        if outcome.rank is None:
            marker = "MANQUE"
        elif outcome.rank == 1:
            marker = "OK   "
        else:
            marker = f"OK@{outcome.rank}"
        print(
            f"  [{marker}] score={outcome.score:.3f} "
            f"top=#{outcome.top_index} ({outcome.top_score:.3f}) [{outcome.question.lang}] "
            f"{outcome.question.text}"
        )
        print(f"          attendu {outcome.question.keywords} -> {outcome.top_snippet}")
    print()

    ranking_report(in_scope)
    refusal_report(out_of_scope, threshold)

    leaks = sum(1 for o in in_scope + out_of_scope if o.leaked)
    print("Etancheite public / membres")
    print("---------------------------")
    print(f"  Fuites non-membre  : {leaks}  {'OK' if leaks == 0 else 'ECHEC'}")
    print()

    all_ranks = [o.rank for o in in_scope if o.rank is not None and o.rank <= max(K_VALUES)]
    return 0 if len(all_ranks) == len(in_scope) and leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
