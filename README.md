<p align="center">
  <img src="docs/logo-full.jpg" alt="Micro Club" width="280">
</p>

<h1 align="center">MicroBot — RAG Assistant for MC Club</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg" alt="FastAPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

**MicroBot** is a retrieval-augmented generation (RAG) service that answers questions about the MC Club (Micro Club) using only trusted, access-controlled documents. It ingests a mixed corpus — PDF, DOCX, Markdown, plain text, Excel/CSV, Google Sheets and web pages — indexes it into a local vector store + lexical (BM25) index, and produces grounded, cited answers gated by a public/members visibility policy.

---

## Highlights

| Capability | Details |
| --- | --- |
| **Multilingual answers** | Arabic, French and English — the assistant answers in the language of the question. |
| **Hybrid retrieval** | Vector search (Chroma + multilingual-e5) fused with BM25 (SQLite FTS5) via Reciprocal Rank Fusion. |
| **Query rewriting** | Weak follow-ups are rewritten into standalone queries with a single extra LLM call. |
| **Multi-query** | Alternative phrasings are searched in parallel and fused for better recall. |
| **Grounded & cited** | Every answer cites its sources `[1] [2]`; if the corpus has no answer, the bot says so. |
| **Access control** | Public vs. members-only documents are filtered **inside each index**, never in the prompt. |
| **Rich ingestion** | PDF, DOCX, Markdown, TXT, XLSX, CSV, Google Sheets (via export URL), web pages (SSRF-guarded). |
| **Admin API** | Upload, replace and delete documents; manage the corpus over HTTP. |
| **Web UI** | Built-in chat interface with document management, rate-limited and token-authenticated. |
| **Evaluation harness** | Offline Hit@1 / MRR / refusal metrics against a labeled dataset (`scripts/evaluate.py`). |

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
  User / Bot ───►│  FastAPI  (/api/v1/*)                        │
                 │  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
                 │  │  /chat  │  │/documents│  │ /feedback   │  │
                 │  └────┬────┘  └────┬─────┘  └─────────────┘  │
                 └───────┼────────────┼─────────────────────────┘
                         ▼            ▼
              ┌────────────────┐  ┌──────────────────┐
              │  ChatService   │  │ IngestionService │
              │  (pipeline)    │  │ (parse→chunk→    │
              └───────┬────────┘  │  embed→index)    │
                      │           └────────┬─────────┘
           ┌──────────┼──────────┐         │
           ▼          ▼          ▼         ▼
     ┌──────────┐ ┌────────┐ ┌──────┐ ┌────────────┐
     │ Retriever│ │  LLM   │ │Parser│ │ Embeddings │
     │ vector + │ │(Gemini │ │ PDF  │ │ (e5-base,  │
     │ BM25 +   │ │/Open-  │ │ DOCX │ │  local)    │
     │ rewriting│ │Router) │ │ XLSX │ └─────┬──────┘
     └────┬─────┘ └────────┘ │ CSV  │       │
          │                  │ Web  │       ▼
          ▼                  └──┬───┘ ┌──────────┐
   ┌─────────────┐              │     │  Chroma   │
   │  Grounding  │◄─────────────┘     │ (persistent
   │  + Citations│                    │  local)   │
   └─────────────┘                    └───────────┘
                                             │
                                      ┌──────┴──────┐
                                      │  BM25 FTS5  │
                                      │  (SQLite)   │
                                      └─────────────┘
```

### Project layout

```
src/mc_rag/
├── main.py                  # FastAPI app factory
├── config.py                # Settings (pydantic-settings, .env)
├── api/                     # HTTP layer: routes, schemas, errors, auth deps
├── domain/                  # Core models, access policy, exceptions
├── application/             # ChatService, IngestionService, ConversationService
├── rag/                     # Retrieval, chunking, grounding, prompts, hybrid, rewriting
└── infrastructure/          # Parsers, embeddings, vector store, lexical store, LLM, Discord, persistence
```

---

## Quick start

### 1. Install

```bash
git clone https://github.com/trrayane/MicroBot.git
cd MicroBot
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Purpose |
| --- | --- |
| `ADMIN_API_KEY` | Key for document-management endpoints (`X-Admin-Key`). |
| `MEMBER_TOKEN` | Bearer token that unlocks members-only content. |
| `GEMINI_API_KEY` or `OPENROUTER_API_KEY` | LLM provider (Gemini takes precedence). |

The embedding model (`intfloat/multilingual-e5-base`) downloads automatically on first run.

### 3. Run

```bash
uvicorn mc_rag.main:app --reload
# → http://127.0.0.1:8000  (web UI at /)
```

### 4. Docker (optional)

```bash
docker compose up --build
```

> Chroma runs **embedded** (persistent client, no separate server). The `storage/` directory holds the vector index, BM25 index and SQLite databases.

---

## API overview

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | — | Liveness probe. |
| `POST` | `/chat` | optional `Bearer` | Ask a question; returns grounded, cited answer. |
| `GET` | `/documents` | `X-Admin-Key` | List indexed documents. |
| `POST` | `/documents` | `X-Admin-Key` | Ingest a file, URL or pasted text. |
| `DELETE` | `/documents/{id}` | `X-Admin-Key` | Remove a document and its chunks. |
| `POST` | `/feedback` | — | Thumbs up/down on an answer. |
| `GET` | `/feedback` | `X-Admin-Key` | List feedback entries. |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qui est le trésorier du bureau ?"}'
```

Response:

```json
{
  "answer": "Le trésorier est Amine [1].",
  "grounded": true,
  "sources": [
    {"doc_id": "…", "title": "Bureau_2025", "snippet": "…"}
  ],
  "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 28},
  "latency_ms": 842
}
```

---

## Retrieval modes

Hybrid retrieval and reranking are **on by default**. Query rewriting and multi-query are off by default; each can be changed with its own environment flag:

| Mode | Flag | What it does |
| --- | --- | --- |
| Hybrid search | `HYBRID_ENABLED=false` disables it | BM25 (SQLite FTS5) **and** Chroma candidates fused with Reciprocal Rank Fusion, then gated by an accept rule. |
| Query rewriting | `QUERY_REWRITE_ENABLED=true` | When the first pass looks weak (or the question is a follow-up), one LLM call rewrites it into a standalone query. |
| Multi-query | `MULTI_QUERY_ENABLED=true` | The same single LLM call also returns alternative phrasings; all queries are searched and fused. |

**LLM budget:** at most **two** calls per question (one rewriting + one answer). A `429` is never retried.

**Visibility:** the `public` / `members` filter is applied **inside each index** (vector and lexical), never in the prompt.

---

## Ingestion pipeline

```
File / URL / Text
      │
      ▼
   Parser  ──►  ParsedDocument (text + optional tabular rows)
      │
      ▼
  Chunker  ──►  Chunks (char-based, tabular rows kept whole)
      │
      ▼
 Embedder  ──►  Vectors (multilingual-e5-base, 768-d, local)
      │
      ▼
   Index   ──►  Chroma (vectors)  +  BM25 FTS5 (lexical)
```

Supported formats: **PDF, DOCX, Markdown, TXT, XLSX, CSV, Google Sheets, HTTP(S) pages** (SSRF-guarded). Excel/CSV rows are rendered as `Header : Value` pairs so each row stays one retrievable unit.

---

## Development

```bash
pytest                          # full test suite (unit + integration + security)
pytest tests/unit -q            # fast unit tests
ruff check .                    # lint
mypy src                        # strict type-check
python scripts/evaluate.py      # offline retrieval evaluation
python scripts/evaluate.py --hybrid   # with BM25 + RRF fusion
python scripts/rebuild_index.py       # rebuild vector + BM25 indexes
```

---

## Configuration reference

See [`.env.example`](.env.example) for every tunable: LLM providers, embedding model, retrieval thresholds, hybrid/rewriting/multi-query knobs, chunking sizes, auth tokens and CORS.

Key thresholds (calibrate with `scripts/evaluate.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RETRIEVAL_MIN_SCORE` | `0.79` | Cosine threshold (English). |
| `RETRIEVAL_MIN_SCORE_MULTILINGUAL` | `0.76` | Cosine threshold (French/Arabic). |
| `HYBRID_BM25_MIN_SCORE` | `1.0` | BM25 floor for lexical candidates. |
| `QUERY_REWRITE_MIN_SCORE` | `0.0` | Below this, trigger a rewrite (0 = use language threshold). |

---

## Roadmap

- [ ] Cross-encoder reranker (e.g. `bge-reranker-v2-m3`) to replace brittle absolute thresholds
- [ ] Rate limiting on `/chat` and `/feedback`
- [ ] Scheduled refresh for URL and Google Sheets sources
- [ ] Parent-document retrieval for long chunks
- [ ] Text-to-SQL over indexed tabular data
- [ ] Temporal awareness (document dates, recency weighting)

---

## License

MIT — see [LICENSE](LICENSE).

---

---

<p align="center">
  <img src="docs/logo-icon.jpg" alt="Micro Club" width="64">
</p>

<p align="center">Built for <b>Micro Club (MC)</b> — Université des Sciences et de Technologie Houari Boumediene.</p>
