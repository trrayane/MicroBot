# mc-club-rag

Retrieval-augmented generation (RAG) service for MC Club content, built with
Python and FastAPI. It ingests a mixed corpus (PDF, DOCX, Markdown, web),
indexes it into a local vector store, and answers questions with grounded,
cited responses gated by access policy (public vs. members-only).

## Project layout

```
mc-club-rag/
├── src/mc_rag/
│   ├── main.py            # FastAPI entrypoint
│   ├── config.py          # Settings from environment
│   ├── api/               # HTTP layer (routes, schemas, errors, deps)
│   ├── domain/            # Core models, access policy, exceptions
│   ├── application/       # Use-case services
│   ├── rag/               # Retrieval, chunking, grounding, citations, prompts
│   └── infrastructure/    # Parsers, embeddings, vector store, LLM, Discord, persistence
├── data/                  # Raw corpus + evaluation datasets
├── storage/               # Chroma index + SQLite database
├── scripts/               # Ingestion, index rebuild, evaluation CLIs
├── tests/                 # unit / integration / api / security / evaluation
└── journals/              # Team journals
```

## Getting started

```bash
cp .env.example .env        # then fill in secrets
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
uvicorn mc_rag.main:app --reload
```

## Common tasks

```bash
python scripts/ingest_corpus.py   # ingest raw corpus into the index
python scripts/rebuild_index.py   # rebuild the vector index
python scripts/evaluate.py        # run the evaluation harness
pytest                            # run the test suite
ruff check . && mypy src          # lint + typecheck
```
