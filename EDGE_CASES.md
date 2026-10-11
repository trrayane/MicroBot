# Edge Cases

Track tricky inputs and boundary conditions the RAG service must handle.

## Ingestion & parsing
- [ ] Empty or zero-byte documents
- [ ] Password-protected / corrupted PDFs
- [ ] Scanned PDFs with no extractable text (OCR required?)
- [ ] Documents exceeding size limits
- [ ] Mixed-language content
- [ ] Duplicate documents (dedup / re-index behavior)
- [ ] Web pages that are JS-rendered or return non-200

## Retrieval & generation
- [ ] Query with no relevant chunks (empty retrieval)
- [ ] Query exceeding embedding max token length
- [ ] Conflicting information across sources
- [ ] Prompt injection embedded in source documents
- [ ] Model returns citations to non-existent chunks
- [ ] LLM timeout / provider rate limiting

## Access control
- [ ] Member-only content requested by an unauthenticated user
- [ ] Expired Discord membership or revoked guild access
- [ ] Token expiry mid-conversation
- [ ] Cross-user conversation access attempts

## Persistence & ops
- [ ] Database schema migration on existing data
- [ ] Vector store unavailable / partial index
- [ ] Concurrent ingestion writes
- [ ] Disk full during index build
