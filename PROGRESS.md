# Progress

## Status: feature-complete (retrieval upgrades)

High-level milestones for the mc-club-rag project.

- [x] Repository structure and module scaffolding
- [x] Configuration and environment placeholders
- [x] Core domain models and access policy
- [x] Ingestion pipeline (parsers, chunking, embeddings)
- [x] Vector store integration (Chroma)
- [x] RAG pipeline (retrieval, grounding, citations)
- [x] LLM integration (OpenRouter / Gemini)
- [x] API routes (chat, auth, documents, feedback, health)
- [x] Discord OAuth and membership verification
- [x] Persistence layer and migrations
- [x] Test suites (unit, integration, api, security, evaluation)
- [x] Evaluation harness and baseline metrics
- [x] Tabular ingestion (XLSX / CSV / Google Sheets / DOCX tables)
- [x] Hybrid search (BM25 SQLite FTS5 + vectors, fused with RRF)
- [x] Conditional query rewriting + multi-query (one shared LLM call)
- [x] Evaluation harness extended with `--hybrid` / `--legacy` modes
- [ ] Containerization and deployment
