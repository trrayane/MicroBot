"""CLI script: retrieval-only evaluation harness for the RAG pipeline.

This harness measures the **retrieval** stage of the pipeline and makes
**zero LLM calls**, so it is free and safe to run hundreds of times.

Two modes, mirroring the two production configurations:

* ``--legacy`` (default) - vector search only, exactly the historical
  behaviour of this script;
* ``--hybrid`` - Chroma **and** a SQLite FTS5 (BM25) index, candidates fused
  with RRF and gated by the accept rule of :mod:`mc_rag.rag.hybrid`.

What it does
------------
1. Builds an isolated, in-memory Chroma index (plus a temporary BM25 index in
   hybrid mode) from an embedded labeled corpus (French / English / Arabic
   documents, both ``public`` and ``members``).
2. Runs a question set split into a **calibration** batch (used to tune the
   score threshold) and a **test** batch (never used for tuning).
3. Computes document-level metrics, where the label is the *expected source
   document* (not a chunk id, so labels do not break when chunk size changes):

   - ``Hit@k`` / ``Recall@k`` for k = 1, 3, 5
   - ``MRR`` (mean reciprocal rank of the first relevant document)
   - ``nDCG@5`` (binary relevance; supports multiple relevant documents)
   - refusal accuracy: out-of-scope questions must fall below the threshold
   - public/members tightness: a non-member search must never return a
     ``members`` chunk (checked in **both** indexes).

Usage
-----
::

    python scripts/evaluate.py                     # legacy + full report
    python scripts/evaluate.py --hybrid            # BM25 + vectors fused (RRF)
    python scripts/evaluate.py --hybrid --bm25-min 2.0
    python scripts/evaluate.py --min-score 0.83    # skip calibration
    python scripts/evaluate.py --split test        # report the test batch only
    python scripts/evaluate.py --json results.json # also dump raw results

The question set contains realistic phrasings and typos. Extend ``CORPUS``,
``CALIBRATION_SET`` and ``TEST_SET`` below to grow the evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Relative paths ("./storage/chroma", ".env", ...) resolve against the project
# root, so the script behaves the same wherever it is launched from.
os.chdir(_REPO_ROOT)

import chromadb  # noqa: E402

from mc_rag.config import get_settings  # noqa: E402
from mc_rag.infrastructure.embeddings.local_embedder import get_embedder  # noqa: E402
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index, IndexEntry  # noqa: E402
from mc_rag.infrastructure.parsers.base import ParsedDocument  # noqa: E402
from mc_rag.infrastructure.vector_store.chroma_store import (  # noqa: E402
    VISIBILITY_MEMBERS,
    VISIBILITY_PUBLIC,
    ChromaStore,
    SearchResult,
)
from mc_rag.rag.chunking import chunk_document  # noqa: E402
from mc_rag.rag.retrieval import Retriever  # noqa: E402

K_VALUES = (1, 3, 5)
NDCG_K = 5
CALIBRATION = "calibration"
TEST = "test"

# =====================================================================================
# 1. Labeled corpus (public + members, FR / EN / AR)
# =====================================================================================


@dataclass(frozen=True, slots=True)
class Doc:
    """A corpus document with the visibility that gates its retrieval."""

    doc_id: str
    title: str
    source: str
    text: str
    visibility: str = VISIBILITY_PUBLIC


CORPUS: tuple[Doc, ...] = (
    Doc(
        doc_id="p_membership_fr",
        title="Adhésion & recrutement",
        source="data/uploads/public/adhesion.md",
        text=(
            "Adhésion et recrutement au Micro Club.\n\n"
            "Le Micro Club est un club exclusivement étudiant : seuls les étudiants actuellement "
            "inscrits peuvent candidater à l'adhésion. Les personnes externes ne sont pas éligibles.\n\n"
            "Les inscriptions se font sur la plateforme officielle du club au début de chaque saison. "
            "Selon les besoins, le département IR-HR peut organiser un entretien de sélection avant de "
            "valider la candidature.\n\n"
            "Enfin, la présence sur le serveur Discord officiel du Micro Club est exigée pour être "
            "membre actif. Pour rejoindre, remplissez le formulaire d'adhésion pendant la période de "
            "recrutement puis passez un entretien avec le bureau."
        ),
    ),
    Doc(
        doc_id="p_departments_fr",
        title="Départements et pôles",
        source="data/uploads/public/departements.md",
        text=(
            "Les départements du Micro Club.\n\n"
            "Le club est organisé en plusieurs départements techniques et organisationnels : le "
            "département Informatique / IT, le département Robotique, le département Médias et "
            "Communication, le département Événements, et le département Ressources Humaines (IR-HR).\n\n"
            "Chaque département regroupe des membres autour d'un domaine précis et dispose de son "
            "propre responsable. Le département Robotique travaille sur la conception de robots et la "
            "participation aux compétitions. Le département IT développe les outils numériques du club, "
            "dont le chatbot MicroBot. Le département Médias gère les réseaux sociaux et la communication."
        ),
    ),
    Doc(
        doc_id="p_discord_fr",
        title="Règles du serveur Discord",
        source="data/uploads/public/discord.md",
        text=(
            "Règlement du serveur Discord officiel.\n\n"
            "Article 13 : les membres doivent utiliser un langage correct et respectueux dans tous les "
            "canaux. Le spam et la perturbation des salons sont interdits. Il est interdit de partager "
            "ou de divulguer des informations privées, des médias ou des données personnelles d'autres "
            "membres.\n\n"
            "Les contenus suggestifs, y compris les GIF, emojis ou propos inappropriés, sont prohibés. "
            "Les discussions politiques, religieuses et sujets clivants ne sont pas autorisés. "
            "L'auto-promotion n'est permise que dans les canaux dédiés. Les pseudos, avatars et bannières "
            "doivent rester respectueux. Les activités illégales sont strictement interdites et les liens "
            "dangereux ou malveillants sont proscrits."
        ),
    ),
    Doc(
        doc_id="p_events_fr",
        title="Événements et hackathons",
        source="data/uploads/public/evenements.md",
        text=(
            "Événements, ateliers et hackathons du Micro Club.\n\n"
            "Le club organise régulièrement des hackathons, des ateliers techniques et des conférences. "
            "Le hackathon annuel se déroule sur un week-end et réunit plusieurs équipes autour de défis "
            "de programmation et de robotique.\n\n"
            "Des ateliers d'initiation au développement web, à l'intelligence artificielle et à "
            "l'électronique sont proposés tout au long de l'année. Les dates sont annoncées sur les "
            "réseaux sociaux du club et sur Discord. Un calendrier des événements est publié au début de "
            "chaque saison."
        ),
    ),
    Doc(
        doc_id="p_membership_en",
        title="Membership Requirements",
        source="data/uploads/public/membership.md",
        text=(
            "Membership requirements at MicroClub.\n\n"
            "MicroClub is a student-only club: only currently enrolled students may apply for "
            "membership, and non-students are not eligible.\n\n"
            "Registration happens through MicroClub's official registration platform at the beginning of "
            "each season. Candidates may be asked to complete an interview organized by the IR-HR "
            "department. Being present on the official MicroClub Discord server is also required for "
            "active membership."
        ),
    ),
    Doc(
        doc_id="p_events_en",
        title="Workshops & Hackathons",
        source="data/uploads/public/events.md",
        text=(
            "Workshops, hackathons and events.\n\n"
            "MicroClub runs technical workshops on web development, artificial intelligence and "
            "electronics throughout the year. The club also hosts an annual hackathon where teams "
            "compete over a weekend on programming and robotics challenges, as well as conferences and "
            "guest talks. Event dates are announced on the club's social networks and Discord."
        ),
    ),
    Doc(
        doc_id="p_projects_en",
        title="Student Projects & Competitions",
        source="data/uploads/public/projects.md",
        text=(
            "Student projects and competitions.\n\n"
            "MicroClub members work on student projects across its departments, including software "
            "tools, robotics and media productions. The club takes part in national and international "
            "competitions, such as robotics contests and coding challenges. Students can join a project "
            "team at any point during the season, subject to their department's availability."
        ),
    ),
    Doc(
        doc_id="p_membership_ar",
        title="شروط الانضمام إلى النادي",
        source="data/uploads/public/انضمام.md",
        text=(
            "شروط الانضمام إلى نادي ميكرو كلوب.\n\n"
            "نادي ميكرو كلوب مخصص للطلبة فقط، ولا يُسمح إلا للطلبة المسجّلين حاليًا بالتقدّم للعضوية، "
            "أما غير الطلبة فغير مؤهلين للانضمام. يتم التسجيل عبر المنصة الرسمية للنادي في بداية كل "
            "موسم. قد يُطلب من المترشحين إجراء مقابلة يشرف عليها قسم الموارد البشرية.\n\n"
            "كما يُشترط التواجد على خادم ديسكورد الرسمي للنادي لتكون عضوًا نشطًا."
        ),
    ),
    Doc(
        doc_id="p_activities_ar",
        title="الأنشطة والفعاليات",
        source="data/uploads/public/انشطة.md",
        text=(
            "الأنشطة والفعاليات.\n\n"
            "ينظّم نادي ميكرو كلوب هاكاثونات وورشات تقنية ومحاضرات على مدار السنة. يشمل الهاكاثون "
            "السنوي فرقًا تتنافس خلال عطلة نهاية الأسبوع حول تحديات البرمجة والروبوتيك. تُعلن تواريخ "
            "الفعاليات على شبكات التواصل الخاصة بالنادي وعلى خادم ديسكورد."
        ),
    ),
    Doc(
        doc_id="m_procedures_fr",
        title="Procédures internes",
        source="data/uploads/members/procedures.md",
        visibility=VISIBILITY_MEMBERS,
        text=(
            "Procédures internes (réservé aux membres).\n\n"
            "Toute demande de budget ou de réservation de salle passe par le responsable de département "
            "puis par le bureau. Pour réserver une salle de réunion, adressez une demande écrite au "
            "bureau au moins 48 heures à l'avance en précisant la date, l'heure et le motif. Les clés et "
            "badges d'accès aux locaux sont gérés par IR-HR."
        ),
    ),
    Doc(
        doc_id="m_budget_fr",
        title="Trésorerie et remboursements",
        source="data/uploads/members/tresorerie.md",
        visibility=VISIBILITY_MEMBERS,
        text=(
            "Trésorerie et remboursements (réservé aux membres).\n\n"
            "Le budget annuel est réparti entre les départements. Chaque département dispose d'une "
            "enveloppe allouée en début de saison, dont celle du département IT pour ses outils et "
            "serveurs. Pour obtenir un remboursement, présentez une facture originale au trésorier, qui "
            "valide la dépense après accord du responsable de département. Les remboursements sont "
            "effectués une fois par mois."
        ),
    ),
    Doc(
        doc_id="m_contacts_fr",
        title="Contacts internes du bureau",
        source="data/uploads/members/contacts.md",
        visibility=VISIBILITY_MEMBERS,
        text=(
            "Contacts internes du bureau (réservé aux membres).\n\n"
            "Coordonnées internes du bureau et des responsables de départements. Le président, le "
            "trésorier et les responsables IT, Robotique, Médias et Événements sont joignables via les "
            "canaux privés réservés aux membres sur Discord. Pour toute urgence, contactez le trésorier "
            "pour les questions financières et IR-HR pour les questions d'accès aux locaux."
        ),
    ),
    Doc(
        doc_id="m_schedule_en",
        title="Internal Schedule & Room Booking",
        source="data/uploads/members/schedule.md",
        visibility=VISIBILITY_MEMBERS,
        text=(
            "Internal schedule and room booking (members only).\n\n"
            "The internal meeting room can be booked by members through the board. Send a written "
            "request at least 48 hours in advance with the date, time and purpose. The weekly internal "
            "schedule of department meetings is shared in the members-only channel on Discord. Access "
            "badges are managed by the HR department."
        ),
    ),
    Doc(
        doc_id="m_board_ar",
        title="معلومات داخلية للأعضاء",
        source="data/uploads/members/داخلي.md",
        visibility=VISIBILITY_MEMBERS,
        text=(
            "معلومات داخلية للأعضاء (خاص بالأعضاء فقط).\n\n"
            "يحتوي هذا المستند على أرقام التواصل الداخلية للمكتب ولرؤساء الأقسام، وجدول الاجتماعات "
            "الأسبوعي. لحجز قاعة الاجتماعات، يرجى تقديم طلب كتابي إلى المكتب قبل 48 ساعة على الأقل مع "
            "تحديد التاريخ والوقت والسبب. بطاقات الدخول إلى المقر تُدار من طرف قسم الموارد البشرية."
        ),
    ),
)

# =====================================================================================
# 2. Evaluation sets (labels = expected source document, not chunk)
# =====================================================================================


@dataclass(frozen=True, slots=True)
class Question:
    """One evaluation question. ``relevant`` empty means it must be refused."""

    text: str
    lang: str
    split: str
    relevant: tuple[str, ...] = ()
    is_member: bool = False


CALIBRATION_SET: tuple[Question, ...] = (
    # --- FR public ---
    Question("comment rejoindre le micro club ?", "fr", CALIBRATION, ("p_membership_fr",)),
    Question(
        "quelles sont les conditions pour devenir membre ?", "fr", CALIBRATION, ("p_membership_fr",)
    ),
    Question("kels sont les departement du club ?", "fr", CALIBRATION, ("p_departments_fr",)),
    Question("quels pôles techniques existent ?", "fr", CALIBRATION, ("p_departments_fr",)),
    Question("quelles sont les règles sur discord ?", "fr", CALIBRATION, ("p_discord_fr",)),
    Question("je peux spammer sur le serveur discord ?", "fr", CALIBRATION, ("p_discord_fr",)),
    Question("quand a lieu le prochain hackathon ?", "fr", CALIBRATION, ("p_events_fr",)),
    Question("y a-t-il des ateliers prévus ?", "fr", CALIBRATION, ("p_events_fr",)),
    # --- EN public ---
    Question("how can i join microclub?", "en", CALIBRATION, ("p_membership_en",)),
    Question("what are the membership requirements?", "en", CALIBRATION, ("p_membership_en",)),
    Question("are there workshops or hackathons?", "en", CALIBRATION, ("p_events_en",)),
    Question(
        "tell me about student projects and competitions", "en", CALIBRATION, ("p_projects_en",)
    ),
    # --- AR public ---
    Question("كيف أنضم إلى النادي؟", "ar", CALIBRATION, ("p_membership_ar",)),
    Question("ما هي شروط العضوية في النادي؟", "ar", CALIBRATION, ("p_membership_ar",)),
    Question("ما هي الأنشطة والفعاليات القادمة؟", "ar", CALIBRATION, ("p_activities_ar",)),
    # --- members-only ---
    Question(
        "quelles sont les procédures internes pour réserver une salle ?",
        "fr",
        CALIBRATION,
        ("m_procedures_fr",),
        True,
    ),
    Question(
        "comment se faire rembourser une dépense ?", "fr", CALIBRATION, ("m_budget_fr",), True
    ),
    Question("quels sont les contacts du bureau ?", "fr", CALIBRATION, ("m_contacts_fr",), True),
    Question(
        "how do i book the meeting room internally?", "en", CALIBRATION, ("m_schedule_en",), True
    ),
    Question("ما هي المعلومات الداخلية للأعضاء؟", "ar", CALIBRATION, ("m_board_ar",), True),
    # --- out-of-scope (must be refused) ---
    Question("quel temps fera-t-il demain à Alger ?", "fr", CALIBRATION),
    Question("what is the capital of Japan?", "en", CALIBRATION),
    Question("ما هو سعر البيتكوين اليوم؟", "ar", CALIBRATION),
    Question("quelle est la recette du couscous ?", "fr", CALIBRATION),
    Question("who is the president of the usa?", "en", CALIBRATION),
    Question("كيف أطبخ الكسكس؟", "ar", CALIBRATION),
    Question("quelle est la hauteur de la tour eiffel ?", "fr", CALIBRATION),
    Question("what's the weather in tokyo?", "en", CALIBRATION),
)

TEST_SET: tuple[Question, ...] = (
    # --- FR public ---
    Question("comment postuler au club ?", "fr", TEST, ("p_membership_fr",)),
    Question("faut-il être étudiant pour adhérer ?", "fr", TEST, ("p_membership_fr",)),
    Question("quels sont les différents départements ?", "fr", TEST, ("p_departments_fr",)),
    Question("est-ce que la robotique est un département ?", "fr", TEST, ("p_departments_fr",)),
    Question("c'est quoi les règles du discord ?", "fr", TEST, ("p_discord_fr",)),
    Question(
        "le langage irrespectueux est-il interdit sur discord ?", "fr", TEST, ("p_discord_fr",)
    ),
    Question("quels événements organise le club ?", "fr", TEST, ("p_events_fr",)),
    Question("quand sont les conférences ?", "fr", TEST, ("p_events_fr",)),
    # --- EN public ---
    Question("how do i apply to microclub?", "en", TEST, ("p_membership_en",)),
    Question("do i need to be a student to join?", "en", TEST, ("p_membership_en",)),
    Question("what events do you organize?", "en", TEST, ("p_events_en",)),
    Question("are there coding competitions for students?", "en", TEST, ("p_projects_en",)),
    # --- AR public ---
    Question("ما هي شروط التسجيل في النادي؟", "ar", TEST, ("p_membership_ar",)),
    Question("هل يمكن لغير الطلبة الانضمام؟", "ar", TEST, ("p_membership_ar",)),
    Question("ما هي الفعاليات التي ينظمها النادي؟", "ar", TEST, ("p_activities_ar",)),
    # --- members-only ---
    Question("comment réserver la salle de réunion ?", "fr", TEST, ("m_procedures_fr",), True),
    Question("quel est le budget réservé au département IT ?", "fr", TEST, ("m_budget_fr",), True),
    Question(
        "à qui m'adresser pour le remboursement ?",
        "fr",
        TEST,
        ("m_budget_fr", "m_contacts_fr"),
        True,
    ),
    Question("quels sont les numéros du bureau ?", "fr", TEST, ("m_contacts_fr",), True),
    Question(
        "who do i contact internally for room booking?",
        "en",
        TEST,
        ("m_schedule_en", "m_contacts_fr"),
        True,
    ),
    Question("what is the internal schedule?", "en", TEST, ("m_schedule_en",), True),
    Question("كيف أحجز قاعة الاجتماع؟", "ar", TEST, ("m_board_ar",), True),
    Question("ما هي معلومات التواصل الداخلية؟", "ar", TEST, ("m_board_ar",), True),
    # --- out-of-scope (must be refused) ---
    Question("how do i bake a chocolate cake?", "en", TEST),
    Question("who won the world cup in 2022?", "en", TEST),
    Question("ما هو أفضل فندق في دبي؟", "ar", TEST),
    Question("quelle est la capitale du Japon ?", "fr", TEST),
    Question("how to fix my car engine?", "en", TEST),
    Question("donne-moi un film à regarder ce soir.", "fr", TEST),
    Question("how much is bitcoin worth today?", "en", TEST),
    Question("ما هي عاصمة المغرب؟", "ar", TEST),
    Question("quelle est la recette de la pizza ?", "fr", TEST),
    Question("how do i learn to play guitar?", "en", TEST),
)


# =====================================================================================
# 3. Metrics
# =====================================================================================


@dataclass(slots=True)
class QuestionResult:
    """Retrieval outcome for a single question (in its own access context)."""

    question: Question
    ranked_docs: list[str]
    best_score: float
    member_leak: bool = False

    @property
    def first_relevant_rank(self) -> int | None:
        relevant = set(self.question.relevant)
        for position, doc_id in enumerate(self.ranked_docs, start=1):
            if doc_id in relevant:
                return position
        return None

    def hit_at(self, k: int) -> bool:
        return bool(set(self.ranked_docs[:k]) & set(self.question.relevant))

    def recall_at(self, k: int) -> float:
        relevant = set(self.question.relevant)
        if not relevant:
            return 0.0
        return len(set(self.ranked_docs[:k]) & relevant) / len(relevant)

    def ndcg_at(self, k: int) -> float:
        relevant = set(self.question.relevant)
        if not relevant:
            return 0.0
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position in range(min(k, len(self.ranked_docs)))
            if self.ranked_docs[position] in relevant
        )
        ideal = sum(1.0 / math.log2(position + 2) for position in range(min(k, len(relevant))))
        return dcg / ideal if ideal else 0.0

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1.0 / rank if rank else 0.0


@dataclass(slots=True)
class RankingMetrics:
    """Averaged ranking metrics over the in-scope questions of a split."""

    n: int
    hit: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0


def compute_ranking(results: list[QuestionResult]) -> RankingMetrics:
    """Average Hit/Recall/MRR/nDCG over scored (in-scope) questions."""
    scored = [r for r in results if r.question.relevant]
    if not scored:
        return RankingMetrics(n=0)
    count = len(scored)
    return RankingMetrics(
        n=count,
        hit={k: sum(r.hit_at(k) for r in scored) / count for k in K_VALUES},
        recall={k: sum(r.recall_at(k) for r in scored) / count for k in K_VALUES},
        mrr=sum(r.reciprocal_rank for r in scored) / count,
        ndcg=sum(r.ndcg_at(NDCG_K) for r in scored) / count,
    )


@dataclass(slots=True)
class RefusalMetrics:
    """Refusal accuracy at a given threshold, treating "answer" as the positive class."""

    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        correct = self.true_positive + self.true_negative
        return correct / total if total else 0.0


def compute_refusal(results: list[QuestionResult], threshold: float) -> RefusalMetrics:
    """Classify each question as answered/refused from its best raw score."""
    tp = fp = tn = fn = 0
    for result in results:
        in_scope = bool(result.question.relevant)
        answered = result.best_score >= threshold
        if in_scope and answered:
            tp += 1
        elif in_scope and not answered:
            fn += 1
        elif not in_scope and answered:
            fp += 1
        else:
            tn += 1
    return RefusalMetrics(threshold, tp, fp, tn, fn)


# =====================================================================================
# 4. Harness
# =====================================================================================


def build_store(corpus: tuple[Doc, ...]) -> tuple[ChromaStore, int]:
    """Index *corpus* into an isolated in-memory Chroma collection."""
    embedder = get_embedder()
    store = ChromaStore(client=chromadb.EphemeralClient(), collection_name="mc_eval")
    chunks_total = 0
    for doc in corpus:
        parsed = ParsedDocument(
            text=doc.text,
            source=doc.source,
            format="md",
            title=doc.title,
            metadata={"visibility": doc.visibility, "doc_id": doc.doc_id},
        )
        chunks = chunk_document(parsed, visibility=doc.visibility)
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc.doc_id
        if not chunks:
            continue
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        store.add_chunks(chunks, embeddings)
        chunks_total += len(chunks)
    return store, chunks_total


def build_retriever(
    store: ChromaStore,
    *,
    hybrid: bool,
    top_k: int | None = None,
    min_score: float | None = None,
    bm25_min: float | None = None,
    pool: int | None = None,
    require_both: bool | None = None,
) -> Retriever:
    """A retriever over the eval store (temporary BM25 index when hybrid)."""
    lexical = None
    if hybrid:
        lexical = BM25Index(Path(tempfile.mkdtemp(prefix="mc_eval_bm25_")) / "bm25.sqlite3")
    return Retriever(
        store=store,
        embedder=get_embedder(),
        lexical=lexical,
        top_k=top_k,
        min_score=min_score if min_score is not None else 0.0,  # harness gates itself
        hybrid_enabled=hybrid,
        hybrid_bm25_min_score=bm25_min,
        hybrid_candidate_pool=pool,
        hybrid_require_both=require_both,
    )


def index_lexical(corpus: tuple[Doc, ...], retriever: Retriever) -> int:
    """Fill the BM25 index of *retriever* with the same chunks as Chroma.

    Returns:
        The number of lexical entries indexed (``0`` when hybrid is off).
    """
    if not retriever.hybrid_enabled:
        return 0
    entries: list[IndexEntry] = []
    for doc in corpus:
        parsed = ParsedDocument(
            text=doc.text,
            source=doc.source,
            format="md",
            title=doc.title,
            metadata={"visibility": doc.visibility, "doc_id": doc.doc_id},
        )
        for chunk in chunk_document(parsed, visibility=doc.visibility):
            entries.append(
                IndexEntry(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={**chunk.metadata, "doc_id": doc.doc_id},
                )
            )
    retriever.lexical.upsert(entries)
    return len(entries)


def _ranked_doc_ids(hits: Sequence[SearchResult], limit: int) -> list[str]:
    """Ordered unique ``doc_id`` values from ranked chunk hits."""
    ordered: list[str] = []
    for hit in hits:
        doc_id = str(hit.metadata.get("doc_id") or "")
        if doc_id and doc_id not in ordered:
            ordered.append(doc_id)
    return ordered[:limit]


def run_questions(retriever: Retriever, questions: tuple[Question, ...]) -> list[QuestionResult]:
    """Retrieve for every question, plus a non-member probe to detect leakage."""
    results: list[QuestionResult] = []
    for question in questions:
        retrieval = retriever.retrieve(question.text, is_member=question.is_member)
        hits = retrieval.hits
        leaked = any(str(h.metadata.get("visibility")) == VISIBILITY_MEMBERS for h in hits)
        best = max((h.score for h in hits), default=0.0)
        results.append(
            QuestionResult(
                question=question,
                ranked_docs=_ranked_doc_ids(hits, max(K_VALUES)),
                best_score=best,
                member_leak=leaked,
            )
        )
    return results


def leak_probe(retriever: Retriever, questions: tuple[Question, ...]) -> int:
    """Count non-member searches that return a ``members`` chunk (must stay 0)."""
    leaks = 0
    for question in questions:
        retrieval = retriever.retrieve(question.text, is_member=False)
        if any(
            str(h.metadata.get("visibility")) == VISIBILITY_MEMBERS for h in retrieval.hits
        ):
            leaks += 1
    return leaks


def calibrate_threshold(results: list[QuestionResult], lo: float = 0.5, hi: float = 0.99) -> float:
    """Pick the threshold maximizing refusal accuracy, tie-broken by higher threshold."""
    best_threshold = lo
    best_score = (-1.0, -1.0)
    steps = int(round((hi - lo) * 100)) + 1
    for step in range(steps):
        threshold = round(lo + step / 100.0, 2)
        metrics = compute_refusal(results, threshold)
        candidate = (metrics.accuracy, threshold)
        if candidate > best_score:
            best_score = candidate
            best_threshold = threshold
    return best_threshold


# =====================================================================================
# 5. Reporting
# =====================================================================================


def print_ranking(title: str, metrics: RankingMetrics) -> None:
    print(title)
    print("-" * len(title))
    if not metrics.n:
        print("  (aucune question in-scope)\n")
        return
    hits = "  ".join(f"Hit@{k}={metrics.hit[k]:.3f}" for k in K_VALUES)
    recalls = "  ".join(f"Recall@{k}={metrics.recall[k]:.3f}" for k in K_VALUES)
    print(f"  Questions        : {metrics.n}")
    print(f"  {hits}")
    print(f"  {recalls}")
    print(f"  MRR              : {metrics.mrr:.3f}")
    print(f"  nDCG@{NDCG_K}          : {metrics.ndcg:.3f}")
    print()


def print_refusal(title: str, metrics: RefusalMetrics) -> None:
    print(title)
    print("-" * len(title))
    print(f"  Seuil            : {metrics.threshold:.2f}")
    print(f"  Accuracy         : {metrics.accuracy:.3f}")
    print(f"  In-scope trouvés : {metrics.true_positive}   manqués : {metrics.false_negative}")
    print(
        f"  Hors-sujet refusés: {metrics.true_negative}   acceptés à tort : {metrics.false_positive}"
    )
    print()


def print_failures(results: list[QuestionResult], threshold: float) -> None:
    missed = [r for r in results if r.question.relevant and not r.hit_at(max(K_VALUES))]
    bad_refusal = [r for r in results if bool(r.question.relevant) != (r.best_score >= threshold)]
    if not missed and not bad_refusal:
        return
    print("Échecs notables")
    print("---------------")
    for result in missed:
        got = result.ranked_docs[0] if result.ranked_docs else "(rien)"
        print(
            f"  [manqué] {result.question.text!r} -> attendu {result.question.relevant}, obtenu {got!r}"
        )
    for result in bad_refusal:
        kind = "non refusé" if result.best_score >= threshold else "refusé à tort"
        print(f"  [{kind}] {result.question.text!r} (score={result.best_score:.3f})")
    print()


def to_json(results: list[QuestionResult], threshold: float) -> dict[str, object]:
    return {
        "threshold": threshold,
        "questions": [
            {
                "text": r.question.text,
                "lang": r.question.lang,
                "split": r.question.split,
                "relevant": list(r.question.relevant),
                "is_member": r.question.is_member,
                "ranked_docs": r.ranked_docs,
                "best_score": round(r.best_score, 4),
                "hit_at_5": r.hit_at(5),
            }
            for r in results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Retrieval-only evaluation harness (no LLM calls).",
    )
    parser.add_argument("--split", choices=["all", CALIBRATION, TEST], default="all")
    parser.add_argument(
        "--min-score", type=float, default=None, help="Skip calibration and use this threshold."
    )
    parser.add_argument(
        "--candidates", type=int, default=20, help="Chunks fetched per query (default: 20)."
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write raw per-question results here."
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Vector-only search (historical behaviour); the default without --hybrid.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse Chroma and BM25 candidates with RRF (adds a temporary FTS5 index).",
    )
    parser.add_argument(
        "--bm25-min",
        type=float,
        default=None,
        metavar="SCORE",
        help="Accept a lexical hit when -bm25 >= SCORE (default: settings).",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=None,
        metavar="N",
        help="Candidates fetched per index before fusion (default: settings).",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="A candidate must pass both the vector AND the lexical gate.",
    )
    parser.add_argument(
        "--rewrite-min-score",
        type=float,
        default=None,
        metavar="SCORE",
        help="Report how many questions would trigger a rewrite below SCORE "
        "(0.0 = language threshold); no LLM call is made.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console fix on Windows
        pass

    args = build_parser().parse_args(argv)
    settings = get_settings()
    hybrid = bool(args.hybrid) and not args.legacy

    store, chunks_total = build_store(CORPUS)
    retriever = build_retriever(
        store,
        hybrid=hybrid,
        top_k=args.candidates,
        bm25_min=args.bm25_min,
        pool=args.pool,
        require_both=args.require_both or None,
    )
    lexical_entries = index_lexical(CORPUS, retriever)
    members_docs = sum(1 for doc in CORPUS if doc.visibility == VISIBILITY_MEMBERS)
    print()
    print("Corpus d'évaluation")
    print("-------------------")
    print(f"  Modèle d'embedding : {settings.embedding_model}")
    print(f"  Mode               : {'hybrid (BM25 + RRF)' if hybrid else 'legacy (vecteurs)'}")
    print(
        f"  Documents          : {len(CORPUS)} ({members_docs} membres, {len(CORPUS) - members_docs} publics)"
    )
    print(f"  Chunks indexés     : {chunks_total}")
    if hybrid:
        print(f"  Entrées lexicales  : {lexical_entries}")
    print()

    calib_results = run_questions(retriever, CALIBRATION_SET)
    test_results = run_questions(retriever, TEST_SET)

    if args.min_score is not None:
        threshold = args.min_score
        print(f"Seuil fourni       : {threshold:.2f}\n")
    else:
        threshold = calibrate_threshold(calib_results)
        print(f"Seuil calibré      : {threshold:.2f} (sur {len(CALIBRATION_SET)} questions)\n")

    if args.split in ("all", CALIBRATION):
        print_ranking("Retrieval — calibration", compute_ranking(calib_results))
        print_refusal("Refus — calibration", compute_refusal(calib_results, threshold))
    if args.split in ("all", TEST):
        print_ranking("Retrieval — test", compute_ranking(test_results))
        print_refusal("Refus — test", compute_refusal(test_results, threshold))
        print_failures(test_results, threshold)

    leaks = leak_probe(retriever, CALIBRATION_SET + TEST_SET)
    print("Étanchéité public / membres")
    print("---------------------------")
    print(f"  Fuites non-membre  : {leaks}  {'OK' if leaks == 0 else 'ÉCHEC'}")
    print()

    if args.rewrite_min_score is not None:
        gate = args.rewrite_min_score or threshold
        weak = sum(
            1
            for result in calib_results + test_results
            if result.question.relevant and result.best_score < gate
        )
        total = sum(1 for result in calib_results + test_results if result.question.relevant)
        print("Réécriture de requête (simulation, 0 appel LLM)")
        print("----------------------------------------------")
        print(f"  Seuil simulé       : {gate:.2f}")
        print(f"  Questions faibles  : {weak} / {total} déclencheraient la réécriture")
        print()

    if args.json is not None:
        payload = {
            "threshold": threshold,
            "mode": "hybrid" if hybrid else "legacy",
            "calibration": to_json(calib_results, threshold),
            "test": to_json(test_results, threshold),
            "member_leaks": leaks,
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Résultats écrits dans {args.json}\n")

    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
