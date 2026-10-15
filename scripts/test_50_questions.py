"""50-question RAG evaluation: visitor / member × FR / EN / AR × with/without context.

Usage:
    python scripts/test_50_questions.py                    # all 50 questions
    python scripts/test_50_questions.py --quick            # retrieval-only, no LLM
    python scripts/test_50_questions.py --lang fr          # only French
    python scripts/test_50_questions.py --role visitor     # only visitor
    python scripts/test_50_questions.py --n 10             # first 10
    python scripts/test_50_questions.py --json results.json

Each question is scored:
  ✓ PASS  – grounded answer returned (or correct refusal for out-of-scope)
  ✗ FAIL  – wrong answer, hallucination, or leak (member content as visitor)
  ~ WARN  – answered but not grounded / no citation

Multi-turn questions reuse the previous conversation_id to test context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)
sys.path.insert(0, str(_REPO / "src"))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class Question:
    text: str
    lang: str  # "fr" | "en" | "ar"
    role: str  # "visitor" | "member"
    # True = should get a sourced answer, False = should refuse (out of scope)
    expect_answer: bool = True
    # If set, continue the previous conversation (same id)
    follow_up: bool = False
    tag: str = ""  # short label for the report
    # filled at runtime
    result: str = ""  # PASS / FAIL / WARN
    answer: str = ""
    grounded: bool = False
    sources: int = 0
    latency_ms: int = 0
    error: str = ""
    conversation_id: str | None = None


# ---------------------------------------------------------------------------
# 50 QUESTIONS
# ---------------------------------------------------------------------------
QUESTIONS: list[Question] = [
    # ═══════════════════════════════════════════════════════════════════════
    # FRENCH — VISITOR (public content, no context)           [1–8]
    # ═══════════════════════════════════════════════════════════════════════
    Question("Comment rejoindre le Micro Club ?", "fr", "visitor", tag="adhésion"),
    Question("Quels sont les départements du club ?", "fr", "visitor", tag="départements"),
    Question("Quelles sont les règles du club ?", "fr", "visitor", tag="règles"),
    Question("Quels projets sont sur le GitHub du club ?", "fr", "visitor", tag="projets"),
    Question("Est-ce que le spam est autorisé sur Discord ?", "fr", "visitor", tag="discord"),
    Question("Qui est le président du club ?", "fr", "visitor", tag="bureau"),
    Question("Comment contacter le club ?", "fr", "visitor", tag="contacts"),
    Question("Y a-t-il des formations disponibles ?", "fr", "visitor", tag="formations"),

    # ═══════════════════════════════════════════════════════════════════════
    # ENGLISH — VISITOR (public content, no context)          [9–16]
    # ═══════════════════════════════════════════════════════════════════════
    Question("How can I join MicroClub?", "en", "visitor", tag="membership"),
    Question("What are the membership requirements?", "en", "visitor", tag="requirements"),
    Question("What projects does the club have on GitHub?", "en", "visitor", tag="projects"),
    Question("What are the Discord rules?", "en", "visitor", tag="discord"),
    Question("What departments exist in the club?", "en", "visitor", tag="departments"),
    Question("Is there a code of conduct?", "en", "visitor", tag="conduct"),
    Question("What events does the club organize?", "en", "visitor", tag="events"),
    Question("Who leads the club?", "en", "visitor", tag="leadership"),

    # ═══════════════════════════════════════════════════════════════════════
    # ARABIC — VISITOR (public content, no context)           [17–24]
    # ═══════════════════════════════════════════════════════════════════════
    Question("كيف أنضم إلى نادي مايكرو؟", "ar", "visitor", tag="عضوية"),
    Question("ما هي شروط العضوية في النادي؟", "ar", "visitor", tag="شروط"),
    Question("ما هي أقسام النادي؟", "ar", "visitor", tag="أقسام"),
    Question("ما هي قواعد النادي؟", "ar", "visitor", tag="قواعد"),
    Question("هل هناك مشاريع على GitHub؟", "ar", "visitor", tag="مشاريع"),
    Question("ما هي الفعاليات التي ينظمها النادي؟", "ar", "visitor", tag="فعاليات"),
    Question("كيف يمكنني التواصل مع النادي؟", "ar", "visitor", tag="تواصل"),
    Question("هل هناك دورات تدريبية؟", "ar", "visitor", tag="دورات"),

    # ═══════════════════════════════════════════════════════════════════════
    # FRENCH — MEMBER (deeper / internal content)             [25–32]
    # ═══════════════════════════════════════════════════════════════════════
    Question(
        "Quelles sont les sanctions en cas de mauvais comportement ?",
        "fr", "member", tag="sanctions",
    ),
    Question("Quel est le rôle des alumni dans le club ?", "fr", "member", tag="alumni"),
    Question("Quelles sont les règles de conduite des membres ?", "fr", "member", tag="conduite"),
    Question("Comment fonctionne le mentorat ?", "fr", "member", tag="mentorat"),
    Question("Quelles sont les règles pour les messages privés ?", "fr", "member", tag="DM"),
    Question("Quelle est la politique de tolérance zéro ?", "fr", "member", tag="zéro-tolérance"),
    Question(
        "Quelles sont les règles de communication officielle ?",
        "fr", "member", tag="communication",
    ),
    Question(
        "Que se passe-t-il si un membre désobéit au bureau ?",
        "fr", "member", tag="insubordination",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # ENGLISH — MEMBER (deeper / internal content)            [33–40]
    # ═══════════════════════════════════════════════════════════════════════
    Question("What is the zero-tolerance policy?", "en", "member", tag="zero-tolerance"),
    Question("What are the sanctions for misbehavior?", "en", "member", tag="sanctions"),
    Question("What role do alumni play?", "en", "member", tag="alumni"),
    Question("What are the Discord server rules in detail?", "en", "member", tag="discord"),
    Question("What is the direct messaging conduct policy?", "en", "member", tag="DM"),
    Question("How does the mentorship system work?", "en", "member", tag="mentorship"),
    Question("What happens if a member disobeys the board?", "en", "member", tag="insubordination"),
    Question("What is the public conduct policy?", "en", "member", tag="public-conduct"),

    # ═══════════════════════════════════════════════════════════════════════
    # ARABIC — MEMBER (deeper / internal content)             [41–45]
    # ═══════════════════════════════════════════════════════════════════════
    Question("ما هي عقوبات سوء السلوك؟", "ar", "member", tag="عقوبات"),
    Question("ما هي سياسة عدم التسامح؟", "ar", "member", tag="عدم-التسامح"),
    Question("ما هو دور الخريجين في النادي؟", "ar", "member", tag="خريجون"),
    Question("ما هي قواعد الخادم على Discord بالتفصيل؟", "ar", "member", tag="ديسكورد"),
    Question("كيف يعمل نظام الإرشاد؟", "ar", "member", tag="إرشاد"),

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-TURN (context) — mixed roles & languages          [46–50]
    # ═══════════════════════════════════════════════════════════════════════
    Question(
        "Quels projets sont sur le GitHub du club ?", "fr", "visitor",
        tag="ctx-projets", follow_up=False,
    ),
    Question(
        "Lequel utilise React ?", "fr", "visitor",
        tag="ctx-react", follow_up=True,
    ),
    Question(
        "What are the Discord rules?", "en", "visitor",
        tag="ctx-discord", follow_up=False,
    ),
    Question(
        "And what about direct messages?", "en", "visitor",
        tag="ctx-dm", follow_up=True,
    ),
    Question(
        "ما هي عقوبات سوء السلوك؟", "ar", "member",
        tag="ctx-عقوبات", follow_up=False,
    ),
    Question(
        "وماذا عن الإهانة العامة؟", "ar", "member",
        tag="ctx-إهانة", follow_up=True,
    ),
]


def banner() -> None:
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗
║           MC CLUB RAG — 50-QUESTION EVALUATION SUITE           ║
╚══════════════════════════════════════════════════════════════════╝{RESET}

{DIM}Corpus: Code of Conduct (PDF), GitHub page, bench text, student list (XLSX)
Roles:  visitor (public only) · member (public + members)
Langs:  FR · EN · AR
Modes:  no context · multi-turn follow-up{RESET}
""")


def ask_api(question: Question, base_url: str, token: str | None) -> dict:
    """Call POST /api/v1/chat and return the parsed response."""
    import httpx

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload: dict = {"message": question.text}
    if question.conversation_id:
        payload["conversation_id"] = question.conversation_id

    resp = httpx.post(
        f"{base_url}/api/v1/chat",
        json=payload,
        headers=headers,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def run_question(q: Question, base_url: str, token: str | None, verbose: bool) -> None:
    start = time.perf_counter()
    try:
        data = ask_api(q, base_url, token)
        q.latency_ms = int((time.perf_counter() - start) * 1000)
        q.answer = data.get("answer", "")
        q.grounded = data.get("grounded", False)
        q.sources = len(data.get("sources", []))
        q.conversation_id = data.get("conversation_id") or q.conversation_id

        if not q.expect_answer:
            # Should refuse (out-of-scope). Grounded=False is correct.
            if not q.grounded:
                q.result = "PASS"
            else:
                q.result = "WARN"  # answered something we expected it to refuse
        else:
            if q.grounded and q.sources > 0:
                q.result = "PASS"
            elif q.grounded:
                q.result = "WARN"  # grounded but no source cards
            else:
                q.result = "FAIL"  # refused or hallucinated
    except Exception as exc:
        q.latency_ms = int((time.perf_counter() - start) * 1000)
        q.error = str(exc)[:200]
        q.result = "FAIL"

    if verbose and q.answer:
        print(f"    {DIM}Answer: {q.answer[:200]}...{RESET}")


def color_result(r: str) -> str:
    m = {
        "PASS": f"{GREEN}✓ PASS{RESET}",
        "FAIL": f"{RED}✗ FAIL{RESET}",
        "WARN": f"{YELLOW}~ WARN{RESET}",
    }
    return m.get(r, r)


def color_lang(lang: str) -> str:
    m = {"fr": f"{MAGENTA}FR{RESET}", "en": f"{CYAN}EN{RESET}", "ar": f"{YELLOW}AR{RESET}"}
    return m.get(lang, lang)


def color_role(role: str) -> str:
    return f"{GREEN}visitor{RESET}" if role == "visitor" else f"{MAGENTA}member {RESET}"


def print_question(q: Question, idx: int) -> None:
    ctx = f" {DIM}[ctx]{RESET}" if q.follow_up else ""
    tag = f" {DIM}({q.tag}){RESET}" if q.tag else ""
    print(
        f"  [{idx:02d}] {color_result(q.result)}  "
        f"{color_lang(q.lang)}  {color_role(q.role)}{ctx}{tag}  "
        f"{q.latency_ms:>5}ms  src={q.sources}  grounded={q.grounded}{tag}"
    )
    if q.error:
        print(f"       {RED}Error: {q.error}{RESET}")


def print_summary(questions: list[Question]) -> None:
    total = len(questions)
    passed = sum(1 for q in questions if q.result == "PASS")
    warned = sum(1 for q in questions if q.result == "WARN")
    failed = sum(1 for q in questions if q.result == "FAIL")

    print(f"\n{BOLD}{CYAN}{'═' * 66}{RESET}")
    print(f"{BOLD}  RÉSUMÉ{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 66}{RESET}")
    print(f"  Total:  {total}")
    print(f"  {GREEN}PASS:   {passed}{RESET}")
    print(f"  {YELLOW}WARN:   {warned}{RESET}")
    print(f"  {RED}FAIL:   {failed}{RESET}")
    pct = (passed / total * 100) if total else 0
    bar_len = 40
    filled = int(bar_len * passed / total) if total else 0
    bar = f"{GREEN}{'█' * filled}{RESET}{DIM}{'░' * (bar_len - filled)}{RESET}"
    print(f"\n  [{bar}] {pct:.1f}%")

    # Per-language breakdown
    print(f"\n{BOLD}  Par langue:{RESET}")
    for lang in ("fr", "en", "ar"):
        subset = [q for q in questions if q.lang == lang]
        if not subset:
            continue
        p = sum(1 for q in subset if q.result == "PASS")
        print(f"    {color_lang(lang)}: {p}/{len(subset)}  ({p / len(subset) * 100:.0f}%)")

    # Per-role breakdown
    print(f"\n{BOLD}  Par rôle:{RESET}")
    for role in ("visitor", "member"):
        subset = [q for q in questions if q.role == role]
        if not subset:
            continue
        p = sum(1 for q in subset if q.result == "PASS")
        print(f"    {color_role(role)}: {p}/{len(subset)}  ({p / len(subset) * 100:.0f}%)")

    # Context questions
    ctx_qs = [q for q in questions if q.follow_up]
    if ctx_qs:
        p = sum(1 for q in ctx_qs if q.result == "PASS")
        print(f"\n{BOLD}  Contexte (multi-turn):{RESET} {p}/{len(ctx_qs)}")

    # Latency
    latencies = [q.latency_ms for q in questions if q.latency_ms > 0]
    if latencies:
        avg = sum(latencies) / len(latencies)
        mx = max(latencies)
        print(f"\n{BOLD}  Latence:{RESET} moy={avg:.0f}ms  max={mx}ms")

    # Failures
    failures = [q for q in questions if q.result == "FAIL"]
    if failures:
        print(f"\n{BOLD}{RED}  Échecs:{RESET}")
        for q in failures:
            print(f"    {RED}✗{RESET} [{q.lang}] {q.role} — {q.text[:60]}")
            if q.error:
                print(f"      {DIM}{q.error}{RESET}")
            elif q.answer:
                print(f"      {DIM}{q.answer[:120]}{RESET}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="50-question RAG evaluation")
    parser.add_argument("--quick", action="store_true", help="Retrieval only, no LLM")
    parser.add_argument("--lang", choices=["fr", "en", "ar"], help="Filter by language")
    parser.add_argument("--role", choices=["visitor", "member"], help="Filter by role")
    parser.add_argument("--n", type=int, help="Limit to N questions")
    parser.add_argument("--json", type=str, help="Export results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show answers")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--token", default=None, help="Bearer token for member role")
    args = parser.parse_args()

    banner()

    # Filter questions
    questions = QUESTIONS[:]
    if args.lang:
        questions = [q for q in questions if q.lang == args.lang]
    if args.role:
        questions = [q for q in questions if q.role == args.role]
    if args.n:
        questions = questions[: args.n]

    print(f"{BOLD}Running {len(questions)} questions...{RESET}\n")

    if args.quick:
        run_retrieval_only(questions, args.verbose)
    else:
        run_api_questions(questions, args)

    print_summary(questions)

    if args.json:
        export = [
            {
                "text": q.text,
                "lang": q.lang,
                "role": q.role,
                "follow_up": q.follow_up,
                "tag": q.tag,
                "result": q.result,
                "answer": q.answer,
                "grounded": q.grounded,
                "sources": q.sources,
                "latency_ms": q.latency_ms,
                "error": q.error,
            }
            for q in questions
        ]
        Path(args.json).write_text(json.dumps(export, ensure_ascii=False, indent=2))
        print(f"  {CYAN}→{RESET} Results exported to {args.json}")


def run_api_questions(questions: list[Question], args: argparse.Namespace) -> None:
    """Run questions through the live API."""
    import httpx

    # Health check
    try:
        resp = httpx.get(f"{args.base_url}/api/v1/health", timeout=5.0)
        resp.raise_for_status()
        print(f"  {GREEN}✓{RESET} Server healthy at {args.base_url}")
    except Exception as exc:
        print(f"\n  {RED}✗ Cannot reach server at {args.base_url}{RESET}")
        print(
            f"  {DIM}Start it: python -m uvicorn mc_rag.main:app "
            f"--host 0.0.0.0 --port 8000{RESET}"
        )
        print(f"  {DIM}Error: {exc}{RESET}\n")
        sys.exit(1)

    # Resolve token for member questions
    token = args.token or os.environ.get("MC_RAG_TOKEN")
    if not token:
        # Try the static member token from .env
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("MEMBER_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    if token:
        print(f"  {GREEN}✓{RESET} Member token loaded")
    else:
        print(f"  {YELLOW}!{RESET} No member token — member questions may fail")

    for i, q in enumerate(questions, 1):
        role_label = q.role
        ctx_label = " ctx" if q.follow_up else ""
        short = q.text[:55]
        print(
            f"  [{i:02d}/{len(questions)}] "
            f"{q.lang.upper():2s} {role_label:8s}{ctx_label:4s} {short}...",
            end="", flush=True,
        )

        # For follow-up questions, reuse the conversation_id from the previous question
        if q.follow_up and i > 1:
            prev = questions[i - 2]
            if prev.conversation_id:
                q.conversation_id = prev.conversation_id

        run_question(q, args.base_url, token, args.verbose)
        print(f" {color_result(q.result)} {q.latency_ms}ms")


def run_retrieval_only(questions: list[Question], verbose: bool) -> None:
    """Run questions through the retriever directly (no LLM, fast)."""
    from mc_rag.api.dependencies import get_lexical_index, get_store
    from mc_rag.infrastructure.embeddings.local_embedder import get_embedder
    from mc_rag.infrastructure.vector_store.chroma_store import allowed_visibilities
    from mc_rag.rag.hybrid import (
        accept,
        candidate_from_lexical,
        candidate_from_vector,
        rrf_fuse,
    )

    print(f"  {CYAN}→{RESET} Loading embedder (first time ~28s)...", flush=True)
    embedder = get_embedder()
    store = get_store()
    lex = get_lexical_index()
    print(f"  {GREEN}✓{RESET} Ready\n")

    from mc_rag.config import get_settings
    settings = get_settings()

    for i, q in enumerate(questions, 1):
        is_member = q.role == "member"
        visibilities = allowed_visibilities(is_member=is_member)
        threshold = settings.retrieval_min_score if q.lang == "en" else settings.retrieval_min_score_multilingual

        # Vector search
        vec_hits = store.search_by_text(
            q.text, embedder, top_k=settings.hybrid_candidate_pool, visibilities=visibilities,
        )
        # Lexical search
        lex_hits = lex.search(q.text, visibilities=visibilities, limit=settings.hybrid_candidate_pool)

        # RRF fusion
        rankings = [[candidate_from_vector(h) for h in vec_hits], [candidate_from_lexical(h) for h in lex_hits]]
        fused = rrf_fuse(rankings, k=settings.hybrid_rrf_k)
        kept = [
            item for item in fused
            if accept(item.candidate, threshold=threshold, bm25_min=settings.hybrid_bm25_min_score)
        ][:settings.retrieval_top_k]

        best_score = max((h.score for h in vec_hits), default=None)
        best_bm25 = max((h.bm25_score for h in lex_hits), default=0.0)
        has_answer = bool(kept)

        q.sources = len(kept)
        q.grounded = has_answer
        q.answer = f"[retrieval-only] {len(kept)} hits, best_score={best_score:.3f}" if best_score else "no hits"

        if not q.expect_answer:
            q.result = "PASS" if not has_answer else "WARN"
        else:
            q.result = "PASS" if has_answer else "FAIL"

        lang = detect_language(q.text)
        ctx = " ctx" if q.follow_up else ""
        print(
            f"  [{i:02d}] {color_result(q.result)}  {color_lang(q.lang)}  {color_role(q.role)}{ctx}  "
            f"hits={q.sources}  best_vec={best_score or 0:.3f}  best_bm25={best_bm25:.1f}  threshold={threshold:.2f}"
        )
        if verbose:
            for item in kept[:3]:
                c = item.candidate
                snippet = c.text[:80] if hasattr(c, "text") else ""
                print(f"       {DIM}→ {snippet}...{RESET}")


if __name__ == "__main__":
    main()
