import sys; sys.path.insert(0, "src")
from mc_rag.api.dependencies import get_store, get_lexical_index

store = get_store()
lex = get_lexical_index()

# Code of Conduct
print("=== CODE OF CONDUCT ===")
results = lex.search("code conduct rules behavior", visibilities=("public",), limit=5)
for h in results:
    print(f"--- {h.entry.doc_id} ---")
    print(h.entry.text[:400])
    print()

# GitHub
print("=== GITHUB ===")
results = lex.search("micro club github repositories", visibilities=("public",), limit=5)
for h in results:
    print(f"--- {h.entry.doc_id} ---")
    print(h.entry.text[:400])
    print()

# XLSX
print("=== XLSX ===")
results = lex.search("liste affichage", visibilities=("public",), limit=3)
for h in results:
    print(f"--- {h.entry.doc_id} ---")
    print(h.entry.text[:400])
    print()

# test_admin
print("=== TEST ADMIN ===")
results = lex.search("test admin", visibilities=("public",), limit=3)
for h in results:
    print(f"--- {h.entry.doc_id} ---")
    print(h.entry.text[:400])
    print()
