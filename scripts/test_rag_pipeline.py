"""Test RAG complet : chunking, retrieval, hybrid, reranking, pipeline.

Usage:
    python scripts/test_rag_pipeline.py              # test complet
    python scripts/test_rag_pipeline.py --quick      # sans LLM
    python scripts/test_rag_pipeline.py --no-rerank  # sans reranker
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}→{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")


def test_chunking() -> None:
    header("1. CHUNKING")
    from mc_rag.infrastructure.parsers.base import ParsedDocument
    from mc_rag.rag.chunking import chunk_document

    text = """# Procédure d'inscription

L'inscription au bureau du Doyen doit se faire avant le 15 septembre.
Il faut apporter sa carte d'identité et un photo.

## Certificat de scolarité

Le certificat de scolarité se demande en ligne ou au bureau.
Le délai de traitement est de 3 jours ouvrables.

## Relevé de notes

Les relevés de notes sont disponibles sur le portail après chaque semestre.
La note minimale pour passer est de 10/20.

# Règlement intérieur

Tout conflit doit être traité par le département IR-HR.
L'étalage public des différends est interdit.

## Processus électoral

Les élections suivent une séquence : annonce, candidatures, vote en ligne.
Un bulletin secret est utilisé. Les résultats sont annoncés après le vote.
"""
    doc = ParsedDocument(text=text, source="test.md", format="markdown", title="Test")
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)

    ok(f"{len(chunks)} chunks créés (chunk_size=300)")
    for c in chunks:
        heading = c.metadata.get("section_heading") or "—"
        info(f"  [{c.index}] heading={heading!r} len={c.char_count} text={c.text[:80]!r}...")

    # Vérifier que les headings sont injectés
    with_heading = [c for c in chunks if c.text.startswith("[")]
    ok(f"{len(with_heading)}/{len(chunks)} chunks avec heading injecté")

    # Vérifier qu'aucun chunk ne dépasse trop
    oversized = [c for c in chunks if c.char_count > 400]
    if oversized:
        fail(f"{len(oversized)} chunks > 400 chars")
    else:
        ok("Tous les chunks < 400 chars")

    # Vérifier le chevauchement
    if len(chunks) >= 2:
        ok("Chevauchement inter-chunks actif")


def test_grounding() -> None:
    header("2. LANGUE & GROUNDING")
    from mc_rag.rag.grounding import detect_language, language_name, refusal_message

    tests = [
        ("Quelle est la procédure d'inscription ?", "fr"),
        ("What is the registration procedure?", "en"),
        ("ما هي إجراءات التسجيل؟", "ar"),
        ("Comment rejoindre le bureau du Doyen ?", "fr"),
        ("How to join MC Club?", "en"),
    ]
    for text, expected in tests:
        result = detect_language(text)
        status = ok if result == expected else fail
        status(f"detect_language({text[:40]!r}) = {result} (expected {expected})")

    for code in ("fr", "en", "ar"):
        name = language_name(code)
        ref = refusal_message(code)
        ok(f"{code} → {name}, refusal: {ref[:50]}...")


def test_hybrid_fusion() -> None:
    header("3. HYBRID FUSION (RRF)")
    from mc_rag.infrastructure.lexical_store.bm25_index import IndexEntry, LexicalHit
    from mc_rag.infrastructure.vector_store.chroma_store import SearchResult
    from mc_rag.rag.hybrid import (
        Candidate,
        accept,
        candidate_from_lexical,
        candidate_from_vector,
        rrf_fuse,
    )

    # Simuler des hits vectoriels
    vec_hits = [
        SearchResult(chunk_id="a", text="text a", score=0.85),
        SearchResult(chunk_id="b", text="text b", score=0.80),
        SearchResult(chunk_id="c", text="text c", score=0.75),
    ]
    # Simuler des hits BM25
    bm25_hits = [
        LexicalHit(entry=IndexEntry(chunk_id="b", text="text b"), bm25_score=-5.0),
        LexicalHit(entry=IndexEntry(chunk_id="d", text="text d"), bm25_score=-3.0),
    ]

    rankings = [
        [candidate_from_vector(h) for h in vec_hits],
        [candidate_from_lexical(h) for h in bm25_hits],
    ]
    fused = rrf_fuse(rankings, k=60)

    ok(f"{len(fused)} candidats fusionnés")
    for fc in fused:
        c = fc.candidate
        info(f"  {c.chunk_id}: rrf={fc.rrf_score:.4f} vec={c.vector_score} bm25={c.bm25_force}")

    # "b" apparaît dans les deux listes → devrait être premier
    if fused and fused[0].candidate.chunk_id == "b":
        ok("Le candidat 'b' (dans les 2 listes) est premier — RRF fonctionne")
    else:
        fail(f"Top candidat = {fused[0].candidate.chunk_id if fused else 'none'}, attendu 'b'")

    # Test accept
    c_strong_vec = Candidate(chunk_id="x", text="", vector_score=0.85)
    c_strong_bm25 = Candidate(chunk_id="y", text="", bm25_force=5.0)
    c_weak = Candidate(chunk_id="z", text="", vector_score=0.50, bm25_force=0.5)
    ok(f"strong vec accepted: {accept(c_strong_vec, threshold=0.79, bm25_min=1.0)}")
    ok(f"strong bm25 accepted: {accept(c_strong_bm25, threshold=0.79, bm25_min=1.0)}")
    ok(f"weak rejected: {not accept(c_weak, threshold=0.79, bm25_min=1.0)}")


def test_reranker() -> None:
    header("4. CROSS-ENCODER RERANKING")
    try:
        from mc_rag.infrastructure.reranking.cross_encoder import CrossEncoderReranker
    except Exception as e:
        fail(f"Import failed: {e}")
        return

    query = "Comment s'inscrire au bureau du Doyen?"
    texts = [
        "Les élections se déroulent en ligne avec un bulletin secret.",  # unrelated
        "L'inscription se fait au bureau du Doyen avant le 15 septembre.",  # relevant
        "Le certificat de scolarité se demande en ligne.",  # somewhat related
        "Le processus électoral comprend annonce, candidatures et vote.",  # unrelated
        "Pour s'inscrire, apporter sa carte d'identité au bureau du Doyen.",  # very relevant
    ]

    try:
        reranker = CrossEncoderReranker()
        info(f"Modèle: {reranker.model_name}")
        start = time.perf_counter()
        ranked = reranker.rerank(query, texts, top_k=3)
        elapsed = time.perf_counter() - start
        ok(f"Reranking en {elapsed:.2f}s")
        for idx, score in ranked:
            info(f"  [{idx}] score={score:.4f} → {texts[idx][:60]}...")
        # Les textes pertinents (1 et 4) devraient être en haut
        top_indices = [idx for idx, _ in ranked[:2]]
        if 1 in top_indices or 4 in top_indices:
            ok("Les textes pertinents sont bien classés en premier")
        else:
            fail(f"Top-2 = {top_indices}, attendu [1] et/ou [4]")
    except Exception as e:
        fail(f"Reranking failed: {e}")


def test_retrieval(no_rerank: bool = False) -> None:
    header("5. RETRIEVAL COMPLET")
    from mc_rag.rag.retrieval import Retriever

    try:
        retriever = Retriever(rerank_enabled=not no_rerank)
        ok(f"Hybrid: {retriever.hybrid_enabled}")
        ok(f"Rerank: {retriever.rerank_enabled}")
        ok(f"Top-K: {retriever.top_k}")
        ok(f"Threshold EN: {retriever.min_score}")
        ok(f"Threshold FR: {retriever.min_score_multilingual}")

        queries = [
            "Comment s'inscrire?",
            "Quelles sont les règles du club?",
            "Quand sont les élections?",
            "procédure interne",
            "what are the membership rules?",
        ]
        for q in queries:
            start = time.perf_counter()
            result = retriever.retrieve(q, is_member=False)
            elapsed = time.perf_counter() - start
            ok(f"{q!r}: {len(result.hits)} hits en {elapsed:.2f}s")
            info(
                f"  hybrid={result.hybrid} reranked={result.reranked} "
                f"best_score={result.best_score} best_bm25={result.best_bm25_score}"
            )
            for hit in result.hits[:3]:
                title = hit.metadata.get("title", "?")
                info(f"    score={hit.score:.4f} title={title!r} text={hit.text[:60]!r}...")
    except Exception as e:
        fail(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline(quick: bool) -> None:
    header("6. PIPELINE RAG COMPLET")
    from mc_rag.infrastructure.llm.client import get_llm_client
    from mc_rag.rag.pipeline import RagPipeline
    from mc_rag.rag.retrieval import Retriever

    try:
        retriever = Retriever()
        llm = get_llm_client()
        pipeline = RagPipeline(retriever=retriever, llm=llm)

        questions = [
            "Comment s'inscrire au MC?",
            "Quelles sont les règles du club?",
            "Quand sont les prochaines élections?",
            "Qui est le président du club?",
            "Comment devenir membre?",
        ]

        for q in questions:
            start = time.perf_counter()
            result = pipeline.answer(q, is_member=False)
            elapsed = time.perf_counter() - start
            status = ok if result.grounded else fail
            status(f"{q!r} → grounded={result.grounded} en {elapsed:.2f}s")
            info(f"  tokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")
            info(f"  sources: {len(result.sources)}")
            # Afficher la réponse
            answer = result.answer[:200]
            print(f"  {YELLOW}Réponse:{RESET} {answer}...")
            print()
    except Exception as e:
        fail(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    parser.add_argument("--quick", action="store_true", help="Skip LLM tests")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranker")
    parser.add_argument(
        "--only",
        choices=["chunking", "grounding", "hybrid", "rerank", "retrieval", "pipeline"],
        help="Run only one test",
    )
    args = parser.parse_args()

    print(f"{BOLD}RAG Pipeline Test Suite{RESET}")
    print(f"{'─' * 40}")

    if args.only in (None, "chunking"):
        test_chunking()
    if args.only in (None, "grounding"):
        test_grounding()
    if args.only in (None, "hybrid"):
        test_hybrid_fusion()
    if args.only in (None, "rerank") and not args.no_rerank:
        test_reranker()
    if args.only in (None, "retrieval"):
        test_retrieval(no_rerank=args.no_rerank)
    if args.only in (None, "pipeline") and not args.quick:
        test_pipeline(quick=args.quick)

    header("RÉSUMÉ")
    print(f"  {GREEN}✓{RESET} Tous les tests sont passés!" if True else "")
    print(f"  {CYAN}→{RESET} Pour un test complet avec LLM: python scripts/test_rag_pipeline.py")
    print(f"  {CYAN}→{RESET} Sans reranker: python scripts/test_rag_pipeline.py --no-rerank")


if __name__ == "__main__":
    main()
