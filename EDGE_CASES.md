# Edge Cases

Track tricky inputs and boundary conditions the RAG service must handle.

## Ingestion & parsing
- [x] Empty or zero-byte documents (raises `EmptyDocumentError`)
- [x] Password-protected / corrupted PDFs (`CorruptDocumentError` / `EncryptedDocumentError`)
- [x] Documents exceeding size limits (`DocumentTooLargeError`)
- [x] Web pages that are JS-rendered or return non-200 (`ParsingError`)
- [x] Excel sheets: merged cells, empty sheets, formula-only cells (no cached value -> empty)
- [x] CSV: delimiter sniffing, BOM / cp1252 fallback
- [x] Google Sheets: unshared / login-required sheet (`ParsingError` with a clear message)
- [x] SSRF: private / CGNAT / metadata IPs blocked before fetching
- [ ] Scanned PDFs with no extractable text (OCR required?)
- [ ] Mixed-language content
- [ ] Duplicate documents (dedup / re-index behavior)

## Retrieval & generation
- [x] Query with no relevant chunks (empty retrieval -> refusal, 0 LLM call)
- [x] Prompt injection embedded in source documents (escaped / fenced in the prompt)
- [x] Model returns citations to non-existent chunks (filtered)
- [x] LLM timeout / provider rate limiting (bounded retries, 429 never retried)
- [x] Weak first pass / follow-up question -> conditional rewrite (1 extra LLM call, capped)
- [x] BM25 candidate below the vector threshold but strong lexically (kept by the accept rule)
- [x] BM25/FTS5 index out of sync with Chroma (non-fatal; `rebuild_index.py` repairs)
- [ ] Query exceeding embedding max token length
- [ ] Conflicting information across sources

## Access control
- [x] Member-only content requested by an unauthenticated user (filtered inside every index)
- [ ] Expired Discord membership or revoked guild access
- [ ] Token expiry mid-conversation
- [ ] Cross-user conversation access attempts

## Persistence & ops
- [ ] Database schema migration on existing data
- [ ] Vector store unavailable / partial index
- [ ] Concurrent ingestion writes
- [ ] Disk full during index build
