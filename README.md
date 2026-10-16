<p align="center">
  <img src="docs/logo-full.jpg" alt="Micro Club" width="280">
</p>

<h1 align="center">MicroBot — Assistant RAG du MC Club</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg" alt="FastAPI"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/Chroma-vector%20store-ff6b6b.svg" alt="Chroma"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

**MicroBot** est un service de génération augmentée par la récupération (RAG) qui répond aux questions sur le MC Club (Micro Club — USTHB) **uniquement à partir de documents de confiance**, avec un contrôle d'accès fin. Il ingère un corpus mixte — PDF, DOCX, Markdown, texte brut, Excel/CSV, Google Sheets et pages web — l'indexe dans un index vectoriel local (Chroma) **et** un index lexical (BM25/SQLite FTS5), puis produit des réponses sourcées et étiquetées, filtrées par une politique de visibilité `public` / `members`.

> **Tout tourne en local** : embeddings (modèle HuggingFace), index vectoriel, index BM25 et bases SQLite. Seul le LLM (Gemini ou OpenRouter) est appelé à l'extérieur. Aucun serveur Chroma séparé n'est nécessaire (mode *embedded*).

---

## Table des matières

1. [Fonctionnalités](#fonctiononalités)
2. [Architecture](#architecture)
3. [Démarrage rapide](#démarrage-rapide)
4. [Authentification & rôles](#authentification--rôles)
5. [API — référence complète](#api--référence-complète)
6. [Web UI](#web-ui)
7. [Pipeline RAG](#pipeline-rag)
8. [Pipeline d'ingestion](#pipeline-dingestion)
9. [Politique d'accès & visibilité](#politique-daccès--visibilité)
10. [Persistance & stockage](#persistance--stockage)
11. [Configuration (.env)](#configuration-env)
12. [Scripts d'évaluation & d'exploitation](#scripts-évaluation--exploitation)
13. [Tests](#tests)
14. [Développement](#développement)
15. [Limites connues & roadmap](#limites-connues--roadmap)
16. [Licence](#licence)

---

## Fonctionnalités

| Capacité | Détails |
| --- | --- |
| **Réponses multilingues** | Arabe, français, anglais — le bot répond dans la langue de la question (détection locale, gratuite). |
| **Recherche hybride** | Vector search (Chroma + `multilingual-e5-base`) fusionné avec BM25 (SQLite FTS5) via **Reciprocal Rank Fusion** (RRF), puis règle d'acceptation. |
| **Reranking cross-encoder** | Réordonnancement des candidats par `cross-encoder/ms-marco-MiniLM-L-6-v2` (pool de 20). |
| **Query rewriting** | Les questions de suivi faibles sont réécrites en requête autonome (1 seul appel LLM partagé avec multi-query). |
| **Multi-query** | Variantes de formulation recherchées en parallèle et fusionnées pour améliorer le rappel. |
| **Réponses ancrées & citées** | Chaque réponse cite ses sources `[1] [2]` ; si le corpus ne contient pas la réponse, le bot le dit (refus prédéfini, sans appel LLM). |
| **Contrôle d'accès** | Documents `public` vs `members` filtrés **dans chaque index** (clause SQL/`where` Chroma), jamais dans le prompt. |
| **Ingestion riche** | PDF, DOCX, Markdown, TXT, XLSX, CSV, Google Sheets (via URL d'export), pages web (garde SSRF par hop). |
| **Ingestion incrémentale & asynchrone** | SHA-256 par fichier, *chunk ids* content-addressés, réutilisation des embeddings inchangés, upload → `202 Accepted` + suivi de progression (`stage`, `chunks_done/total`). |
| **API d'administration** | Uploader, remplacer, supprimer des documents ; lister les questions récurrentes ; statistiques du corpus. |
| **Authentification** | Clé admin (`X-Admin-Key`), token membre statique (`Bearer`), **OAuth Discord** (JWT HS256, whitelist admins par rôles/IDs). |
| **Web UI intégrée** | SPA monofichier (HTML/CSS/JS) : chat, gestion documents, dashboard admin, feedback, compte Discord. |
| **Feedback & analytics** | Votes utile/pas utile couplés aux logs de requêtes ; questions récurrentes côté admin. |
| **Harnais d'évaluation** | Métriques offline Hit@k / MRR / nDCG / refus / étanchéité members, **zéro appel LLM** (`scripts/evaluate*.py`). |

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

### Structure du projet

```
src/mc_rag/
├── main.py                  # App factory FastAPI + lifespan (warmup embeddings)
├── config.py                # Settings pydantic-settings (chargées depuis .env)
├── api/                     # Couche HTTP
│   ├── router.py            # Montage de tous les routers sous /api/v1
│   ├── dependencies.py      # Résolution d'accès (X-Admin-Key / Bearer / JWT)
│   ├── schemas.py           # Tous les modèles Pydantic de requête/réponse
│   ├── errors.py            # Handlers d'erreurs → {"error":{"code","message"}}
│   └── routes/              # health, admin, auth, chat, documents, feedback
├── domain/                  # Modèles du domaine (DiscordMember, exceptions)
├── application/             # Services applicatifs
│   ├── chat_service.py      # Orchestration d'une requête chat
│   ├── ingestion_service.py # Pipeline d'ingestion complet (634 lignes)
│   ├── ingestion_status.py  # Tracker de progression des jobs (in-memory)
│   ├── conversation_service.py  # Historique + isolation members
│   └── authentication_service.py # Flux OAuth Discord → JWT
├── rag/                     # Cœur du RAG
│   ├── pipeline.py          # answer() : retrieval → LLM → grounding → sources
│   ├── retrieval.py         # Recherche, rewriting, gating, reranking
│   ├── hybrid.py            # Fusion RRF + règle d'acceptation
│   ├── query_rewriting.py   # Rewriting + multi-query (1 appel LLM partagé)
│   ├── chunking.py          # Découpage récursif + tabulaire + titres de section
│   ├── grounding.py         # Détection de langue, refus, marqueur [[NO_ANSWER]]
│   ├── citations.py         # Extraction des [1][2] → Source dédupliquées
│   └── prompts.py           # Prompt système + format <sources>
└── infrastructure/
    ├── parsers/             # pdf, docx, markdown, xlsx, csv, web, gsheets, url_safety
    ├── embeddings/          # LocalEmbedder (sentence-transformers, e5-base)
    ├── vector_store/        # ChromaStore (persistant, cosine, filtres visibilité)
    ├── lexical_store/       # BM25Index (SQLite FTS5) + tokenizer multilingue
    ├── reranking/           # CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
    ├── llm/                 # GeminiClient, OpenRouterClient, sélection
    ├── discord/             # OAuth client + vérification membership/guild
    ├── persistence/         # SQLite : conversations, feedback, query_logs, users
    ├── tokens.py            # Émission/décodage JWT HS256
    └── static/index.html    # Web UI (SPA monofichier)
scripts/                     # Évaluation, ingestion CLI, rebuild d'index, tests
tests/                       # unit / integration / security / api
```

---

## Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/trrayane/MicroBot.git
cd MicroBot

# Option A — uv (recommandé, rapide)
uv venv .venv
uv pip install -e ".[dev]" --python .venv/Scripts/python.exe

# Option B — venv classique
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

### 2. Configuration

```bash
cp .env.example .env
```

Éditez `.env` et renseignez au minimum :

| Variable | Rôle |
| --- | --- |
| `ADMIN_API_KEY` | Clé d'administration (header `X-Admin-Key`). |
| `MEMBER_TOKEN` | Token statique membre (header `Authorization: Bearer …`). |
| `GEMINI_API_KEY` **ou** `OPENROUTER_API_KEY` | Fournisseur LLM (Gemini a la priorité). |
| `DISCORD_CLIENT_ID` / `DISCORD_CLIENT_SECRET` / `DISCORD_GUILD_ID` | Optionnel — active l'OAuth Discord. |

En mode dev, les valeurs par défaut du `.env.example` suffisent : `dev-admin-key` / `dev-member-token`.

> Le modèle d'embedding (`intfloat/multilingual-e5-base`, ~440 Mo) se **télécharge automatiquement** au premier démarrage (plusieurs minutes). Le healthcheck Docker utilise un `start_period` de 300 s pour cette raison.

### 3. Lancer

```bash
# avec uvicorn (dev)
uvicorn mc_rag.main:app --reload
# → http://127.0.0.1:8000  (Web UI sur /, docs OpenAPI sur /docs)

# ou via le binaire du venv
.venv\Scripts\python.exe -m uvicorn mc_rag.main:app --host 127.0.0.1 --port 8000
```

### 4. Docker (optionnel)

```bash
docker compose up --build
```

Le `compose.yaml` :
- expose le port **8000** ;
- monte `./data` (uploads + corpus raw) et `./storage` (SQLite + Chroma + BM25) pour survivre aux rebuilds ;
- conserve le modèle HuggingFace dans un volume nommé `hf-cache` ;
- healthcheck `GET /api/v1/health` toutes les 30 s (`start_period: 300s` au 1ᵉʳ boot).

Chroma tourne **embedded** (client persistant in-process, pas de serveur séparé).

---

## Authentification & rôles

L'API accepte **trois mécanismes**, résolus dans cet ordre par `get_access()` (`api/dependencies.py`) :

1. **Header `X-Admin-Key`** (comparaison en temps constant `hmac.compare_digest`) :
   - clé valide → `is_admin=true, is_member=true` ;
   - clé invalide → **401** immédiat (le Bearer n'est même pas regardé).
2. **Header `Authorization: Bearer <token>`** :
   - égal au `MEMBER_TOKEN` statique → membre (pas admin) ;
   - sinon décodé comme **JWT Discord** (HS256) → membre + claims (`admin`, `user_id`, `name`, `avatar`) ;
   - JWT invalide/expiré → silencieusement ignoré (on passe à l'anonyme, sauf sur routes protégées où `require_admin` renverra 401).
3. **Aucune crédentielle** → anonyme (`public` uniquement).

| Rôle | Obtention | Droits |
| --- | --- | --- |
| **Anonymous** | Aucun header | Chat + feedback ; documents `public` uniquement. |
| **Member** | `MEMBER_TOKEN` ou JWT Discord | Documents `public` **et** `members` ; conversations dédiées. |
| **Admin** | `ADMIN_API_KEY` ou JWT avec `admin=true` | Tout + `/documents`, `/stats`, `/admin/*`, `/feedback` (GET), `/auth/users`. |

> Les endpoints protégés lèvent **401** (pas 403) quand le privilège est insuffisant.

### Flux OAuth Discord (optionnel)

1. `GET /api/v1/auth/discord/login` → génère un `state` (TTL 10 min, en mémoire, usage unique) et redirige vers Discord (scopes `identify guilds.members.read`).
2. Discord rappelle `GET /api/v1/auth/discord/callback?code=…&state=…`.
3. Le serveur échange le `code` contre un `access_token` (jamais stocké), appelle `/users/@me` puis `/users/@me/guilds/{guild_id}/member` (**fonctionne sans bot** dans le serveur).
4. Vérification : l'utilisateur doit être dans la guild ; si `DISCORD_MEMBER_ROLE_ID` est défini, le rôle est exigé **sauf pour les admins**. Admin = rôles dans `DISCORD_ADMIN_ROLE_IDS` **ou** `user_id` dans `DISCORD_ADMIN_USER_IDS`.
5. L'utilisateur est persisté (`storage/users.db`) et un **JWT HS256** est émis (`sub`, `name`, `admin`, `iat`, `exp`, `avatar`), durée `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (720 dans l'exemple).
6. Redirection finale `302 → /#token=<JWT>` ; la Web UI lit le fragment, stocke le token et l'envoie en `Authorization: Bearer`.

Erreurs de callback : `/#error=access_denied`, `/#error=invalid_state`, `/#error=not_member`.

### Login admin (Web UI)

`POST /api/v1/auth/login` avec `{"key": "<ADMIN_API_KEY>"}` → `{"ok": true}` ou 401. La UI stocke ensuite la clé dans `sessionStorage` et l'envoie en `X-Admin-Key`.

---

## API — référence complète

Toutes les routes sont préfixées par **`/api/v1`**. OpenAPI interactif : **`/docs`**.

### Enveloppe d'erreur

Toutes les erreurs suivent le format :

```json
{ "error": { "code": "unauthorized", "message": "Admin privileges required." } }
```

| Exception | HTTP | `code` |
| --- | --- | --- |
| `AuthenticationError` | 401 | `unauthorized` |
| `ConversationNotFoundError` | 404 | `conversation_not_found` |
| `InvalidInput` | 422 | `invalid_input` |
| `LLMRateLimitError` | 429 | `rate_limited` |
| `LLMUnavailableError` | 503 | `llm_unavailable` |
| `UpstreamUnavailable` | 503 | `upstream_unavailable` |
| `VectorStoreError` / `EmbeddingError` | 503 | `service_unavailable` |
| Validation Pydantic | 422 | `validation_error` |
| Exception inattendue | 500 | `internal_error` |

### Inventaire des endpoints

| # | Méthode | Chemin | Auth | Description |
| --- | --- | --- | --- | --- |
| 1 | GET | `/health` | — | Liveness (ne touche ni modèle ni index). |
| 2 | GET | `/stats` | admin | `{collection, documents, chunks}`. |
| 3 | GET | `/admin/repetitive-questions` | admin | Questions récurrentes (`?min_frequency=10`). |
| 4 | POST | `/auth/login` | — | Vérifie une clé admin `{"key": "…"}`. |
| 5 | GET | `/auth/discord/login` | config OAuth | Redirection 302 vers Discord. |
| 6 | GET | `/auth/discord/callback` | config OAuth | Callback OAuth → `/#token=` ou `/#error=`. |
| 7 | GET | `/auth/me` | optionnelle | `{role: admin\|member\|anonymous, display_name, avatar, user_id}`. |
| 8 | GET | `/auth/users` | admin | Liste des utilisateurs connus. |
| 9 | POST | `/chat` | optionnelle | Question → réponse sourcée. |
| 10 | GET | `/documents` | admin | Liste des documents indexés. |
| 11 | GET | `/documents/{doc_id}/status` | admin | Progression d'ingestion d'un document. |
| 12 | POST | `/documents` | admin | Ingestion (file/url/text) → **202**. |
| 13 | DELETE | `/documents/{doc_id}` | admin | Supprime document + chunks (Chroma + BM25). |
| 14 | POST | `/feedback` | — | Vote `up`/`down` sur une réponse. |
| 15 | GET | `/feedback` | admin | Liste des votes (`?limit=100`, max 500). |

### Exemples

#### Chat

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qui est le trésorier du bureau ?"}'
```

Réponse :

```json
{
  "answer": "Le trésorier est Amine [1].",
  "conversation_id": "c_ab12cd34ef56",
  "query_log_id": 42,
  "grounded": true,
  "sources": [
    { "doc_id": "…", "title": "Bureau_2025", "snippet": "… Amine | Trésorier …" }
  ],
  "usage": { "llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 28 },
  "latency_ms": 842
}
```

- `message` : 1–2000 caractères.
- `conversation_id` : optionnel — fourni, le serveur renvoie l'historique récent (6 messages) ; omis/`null`, nouvelle conversation.
- `grounded: false` + réponse de refus = rien de pertinent trouvé (**0 appel LLM**).
- Conversations créées par un membre sont **member-only** : un anonyme qui connaît l'ID reçoit 404.

#### Ajouter un document (fichier)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "X-Admin-Key: dev-admin-key" \
  -F "mode=file" \
  -F "visibility=members" \
  -F "title=Règlement 2026" \
  -F "file=@reglement.pdf"
```

→ **202 Accepted** : `{doc_id, title, status: "processing", chunks: 0, visibility}`.  
Poll ensuite `GET /documents/{doc_id}/status` :

```json
{
  "doc_id": "…", "title": "Règlement 2026",
  "status": "processing", "stage": "embedding",
  "chunks_done": 32, "chunks_total": 120,
  "chunks": 0, "error": null, "elapsed_ms": 4210
}
```

`status` terminal : `indexed` (nouveau), `updated` (remplacé), `unchanged` (hash identique, ignoré), `failed`.

Autres modes : `mode=url` (+ champ `url`, Google Sheets ou page web), `mode=text` (+ champ `text`). Fichiers : **10 Mo max**, suffixes `.pdf .docx .md .markdown .txt .xlsx .csv`.

#### Supprimer un document

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/<doc_id> \
  -H "X-Admin-Key: dev-admin-key"
# → { "doc_id": "…", "chunks_deleted": 12 }
```

#### Feedback

```bash
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "c_…", "query_log_id": 42, "vote": "down"}'
```

---

## Web UI

SPA monofichier servie à la racine **`/`** (`src/mc_rag/static/index.html`, HTML/CSS/JS pur, sans build) :

| Vue | Contenu |
| --- | --- |
| **Chat** | Bulles markdown, citations `[n]` en exposant, effet de typing, RTL arabe, boutons Utile/Pas utile, historique local multi-conversations. |
| **Connexion bureau** | Écran de garde : saisie de la clé admin → `POST /auth/login`. |
| **Dashboard** | Santé du service, stats corpus, **questions récurrentes** (fréquence + votes « pas utile », bouton « ajouter à la base »). |
| **Documents** | Ajout en 3 onglets (fichier drag & drop / lien web / texte), sélecteur de visibilité public/membres, table paginée/recherchable, suppression avec confirmation. |
| **Feedback** | Historique des votes. |
| **Compte Discord** | Bouton de connexion OAuth, badge de vérification membre (payload du JWT). |
| **Jobs d'ingestion** | Panneau flottant avec barres de progression par stage (poll adaptatif 1 s → 2 s → 4 s). |

Thème sombre/clair (toggle, persisté `localStorage.mc_theme`). L'UI reconnaît le token OAuth dans le fragment URL (`#token=…`).

---

## Pipeline RAG

### Vue d'ensemble d'une requête `POST /chat`

```
message + conversation_id
   │
   ├─ ConversationService.open()     historique (6 derniers messages, règles members)
   │
   ├─ RagPipeline.answer()
   │    ├─ detect_language()          fr | en | ar  (gratuit, déterministe)
   │    ├─ Retriever.retrieve()
   │    │    ├─ build_search_query()  follow-up ≤ 6 mots → concaténation
   │    │    ├─ Chroma.search_by_text()   top 20, filtre visibilité
   │    │    ├─ BM25.search()             top 20, WHERE visibility IN (…)
   │    │    ├─ QueryRewriter.plan()      0 ou 1 appel LLM (rewrite + variants)
   │    │    ├─ rrf_fuse() + accept()     fusion RRF (k=60) + règle OU/ET
   │    │    └─ CrossEncoder.rerank()     pool 20 → top 5
   │    │
   │    ├─ si aucun hit → refus prédéfini (langue détectée), 0 appel LLM
   │    ├─ build_messages()           system + history[-4:] + <sources>
   │    ├─ LLM.complete()             1 appel (Gemini ou OpenRouter)
   │    ├─ parse_model_answer()       marqueur [[NO_ANSWER]] → grounded ?
   │    └─ build_sources()            citations [n] → Source dédupliquées
   │
   ├─ QueryLogRepository.add_query()  log SQLite normalisé
   └─ ConversationService.add_turn()  persistance user + assistant
```

**Budget LLM garanti** : **au plus 2 appels** par question (1 rewriting optionnel + 1 réponse), **0** si rien trouvé. Un `429` n'est **jamais** réessayé.

### Recherche hybride (BM25 + vecteurs)

- **Canal vectoriel** : Chroma, distance cosinus, score = `1 - distance`. Texte requêté préfixé `"query: "` (modèle e5).
- **Canal lexical** : SQLite FTS5 (`unicode61 remove_diacritics 2`), tokenisation NFKD + casefold (accents retirés, arabe conservé), match `OR` sur ≤ 32 tokens. Score `bm25()` négatif ; le code utilise `force = -bm25`.
- **Fusion RRF** : `score = Σ 1/(60 + rank_i)` sur chaque liste de rangs — les échelles cosinus/BM25 ne se mélangent jamais. Un candidat présent dans les deux listes bat naturellement un candidat d'une seule.
- **Règle d'acceptation** (`accept`) : garder si `vector_score >= seuil` **OU** `force >= 1.0` (config `HYBRID_REQUIRE_BOTH=true` pour un ET strict).
- **Seuils de pertinence** : `0.79` (anglais) / `0.76` (français-arabe, cosinus cross-lingual plus comprimé). Calibrés par `scripts/evaluate.py`.
- Le **reranker cross-encoder** réordonne ensuite le top-20 (seuil logit `-5.0`, donc quasi jamais filtrant — il sert surtout à réordonner).

### Query rewriting & multi-query

- Désactivés par défaut (`QUERY_REWRITE_ENABLED=false`, `MULTI_QUERY_ENABLED=false`).
- Déclenchés si le meilleur score vectoriel < seuil **ou** s'il y a un historique de conversation.
- **Un seul appel LLM** produit à la fois la requête réécrite (≤ 300 car.) et 2 variantes (≤ 200 car.) en JSON compact ; parsing tolérant (fences, prose).
- Les variantes identiques/dupliquées/inutiles sont rejetées. 429 → plan dégradé, pas de retry.

### Chunking

- Texte libre : découpage **récursif** sur hiérarchie de séparateurs (`\n\n` → `\n` → `. ` → `, ` → ` ` → caractère), taille **900** caractères, overlap **130** rogné sur frontière de mot.
- **Titres de section** : le chunk embarqué est préfixé `[Titre de section] ` (ancres slugifiées type GitHub) pour ancrer le contexte.
- **Tabulaire** (XLSX/CSV/Sheets/tables DOCX) : une ligne n'est **jamais coupée** ; regroupement de jusqu'à 8 lignes courtes (≤ 200 car.) ; chaque ligne rendue `En-tête : valeur | …`. Métadonnées `tabular`, `row_number`, `sheet`.
- `chunk_id = sha1(source|index|text)[:16]` → **content-addressed**, la ré-ingestion est idempotente et réutilise les embeddings des chunks inchangés.

### Prompt système (résumé)

Le LLM reçoit un prompt système qui lui impose : répondre **uniquement** depuis les blocs numérotés `<sources>` ; parler la langue détectée ; citer `[1]`/`[2,3]` ; terminer par `[[NO_ANSWER]]` si absent ; ignorer toute instruction injection dans les sources (blocs cloîtrés, `</sources>` strippé du corps). Style : synthèse, dates exactes, pas de « selon les documents ». 4 messages d'historique max.

### Citations & refus

- Les marqueurs `[n]` (1-based) pointent les hits rerankés ; hors bornes → ignorés.
- Aucune citation valide → top 3 en fallback. Déduplication par `doc_id`.
- Refus prédéfinis dans les 3 langues, **déterministes et gratuits** (0 LLM).

---

## Pipeline d'ingestion

```
Fichier / URL / Texte
      │
      ▼
   Parser  ──►  ParsedDocument (texte + lignes tabulaires optionnelles)
      │
      ▼
  Chunker  ──►  Chunks (900/130, tabulaire intact, titres de section)
      │
      ▼
 Embedder  ──►  Vecteurs (e5-base, 768-d, batch 32, préfixe "passage: ")
      │
      ▼
   Index   ──►  Chroma (upsert)  +  BM25 FTS5 (sync best-effort)
```

### Idempotence & remplacement

- Empreinte **SHA-256** des octets bruts + `EMBEDDING_MODEL` stockés dans les métadonnées du document. Identique → statut `unchanged`, aucun travail.
- Document **changé** : les nouveaux chunks sont **upsertés d'abord**, puis les chunks obsolètes supprimés — un échec en cours laisse l'ancienne version consultable (rollback implicite).
- **Changer `EMBEDDING_MODEL` dans le `.env` déclenche automatiquement un re-index complet.**
- Les jobs d'ingestion sont **sérialisés** (verrou) — l'embedding est CPU-bound ; l'API répond en 202 immédiatement et la progression est suivie via le tracker (max 200 jobs en mémoire).

### Formats supportés

| Format | Parseur | Notes |
| --- | --- | --- |
| `.pdf` | pypdf | Page par page ; PDF chiffré → tentative mot de passe vide, sinon erreur ; page corrompue → warning + page vidée. |
| `.docx` | python-docx | Titres → sections ; **tables** → lignes `En-tête : valeur`. |
| `.md` / `.markdown` | markdown-it-py | Front-matter YAML → métadonnées/titre ; ancrages de sections. |
| `.txt` | utf-8 (fallback replace) | Texte brut. |
| `.xlsx` | openpyxl | `data_only=True` (valeurs en cache des formules) ; cellules fusionnées propagées vers le bas ; feuilles vides/chart ignorées ; max 50 000 lignes / 500 car. par cellule. |
| `.csv` | stdlib csv | Sniffing du délimiteur (`; , \t |`) ; utf-8-sig → cp1252 → replacement. |
| Google Sheets | export xlsx/csv | URL publique uniquement (`/spreadsheets/d/…`) ; 401/403 ou page de login → erreur claire. |
| Page web `http(s)` | httpx + BeautifulSoup | Scripts/styles/nav strippés ; **garde SSRF** (voir ci-dessous). |

### Garde SSRF (pages web & Sheets)

- Uniquement `http`/`https`, pas de credentials dans l'URL.
- Toutes les IPs résolues doivent être **publiques** : loopback, privées, link-local, CGNAT `100.64/10`, `192.0.0/24`, `198.18/15` et la metadata cloud `169.254.169.254` sont bloquées.
- Redirections suivies **manuellement** (5 max), chaque hop revalidé. Timeout 10 s.
- `allow_local_files=False` sur l'endpoint URL : impossible de lire un fichier local via une « URL ».

### Upload API

- Écriture **streamée** (blocs de 1 MiB), plafond **10 Mo** (413 sinon, fichier partial supprimé).
- Nom de fichier assaini (`[^A-Za-z0-9._-]` → `_`).
- Propriétaire enregistré (`storage/users.db` → champ `uploaded_by` de `GET /documents`).
- Fix notable : `UserStore` utilise un `RLock` **réentrant** — un `Lock` simple auto-deadlockait le premier upload (la property `_connection` reprenait le lock déjà tenu).

---

## Politique d'accès & visibilité

Deux visibilités : **`public`** et **`members`**.

- Le filtre s'applique **dans chaque index** :
  - Chroma : clause `where` `{"visibility": {"$in": [...]}}` combinée en `$and` (impossible d'élargir son accès par un `where` supplémentaire) ;
  - BM25 : `WHERE c.visibility IN (…)` en SQL.
- Par défaut sur un `search()` sans filtre : **public uniquement**.
- À l'ingestion : le **premier segment du chemin** sous `data/raw/` (`data/raw/members/…`) fixe la visibilité ; sinon `public`. L'argument explicite de l'API gagne toujours.
- Conversations : une conversation ouverte par un membre devient **member-only** ; un anonyme qui tient l'ID reçoit 404. Une conversation anonyme reprise par un membre est **upgradée** member-only.

Ces règles sont couvertes par `tests/security/test_visibility_filtering.py` (fuite members = 0 exigée par les harnais d'éval).

---

## Persistance & stockage

Tout vit sous `./storage/` et `./data/` (bind-mountés en Docker) :

| Fichier | Contenu |
| --- | --- |
| `storage/chroma/` | Index vectoriel Chroma (collection `micro_bot_corpus_e5base`, espace cosinus). |
| `storage/bm25.sqlite3` | Index lexical FTS5 (source de vérité secondaire, réparable via `rebuild_index.py`). |
| `storage/conversations.db` | Conversations + messages (`c_<id>`, rôles user/assistant). |
| `storage/feedback.db` | Votes utile/pas utile. |
| `storage/users.db` | Utilisateurs Discord + propriétaires de documents. |
| `storage/uploads/` | Fichiers uploadés (staging). |
| `data/raw/public/`, `data/raw/members/` | Corpus raw pour l'ingestion CLI (`ingest_corpus.py`). |

Les `query_logs` (normalisation NFKC+casefold+ponctuation → espaces) vivent dans `conversations.db` et alimentent le dashboard (questions récurrentes, `unuseful_count`).

> `DATABASE_URL` (`mc_club.db`) existe dans la config (SQLAlchemy/Alembic sont des dépendances) mais la persistance réelle utilise le **`sqlite3` stdlib** dans les dépôts ci-dessus.

---

## Configuration (.env)

Voir [`.env.example`](.env.example) pour la liste exhaustive. Résumé des plus importantes :

### Application

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | Bind uvicorn. |
| `APP_ENV` | `development` | Environnement (affiché par `/health`). |
| `LOG_LEVEL` | `INFO` | Niveau de log. |
| `CORS_ORIGINS` | `localhost:3000,localhost:8000` | Origines CORS (CSV). |

### LLM

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Prioritaire** si présente (alias `GOOGLE_API_KEY`). |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Modèle Gemini natif. |
| `OPENROUTER_API_KEY` | — | Secours si pas de Gemini. |
| `OPENROUTER_MODEL` | `google/gemma-4-31b-it:free` | Modèle OpenRouter. |

Timeout 45 s ; OpenRouter : 2 tentatives max (réseau/5xx uniquement), **jamais sur 429**. Pas de bascule dynamique entre providers : le choix est fait au démarrage (Gemini gagne s'il est présent).

### Embeddings & index

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modèle local (changement ⇒ re-index auto). |
| `EMBEDDING_DIMENSION` | `768` | Dimension (contrôlée au chargement). |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch d'encodage. |
| `EMBEDDING_QUERY_PREFIX` / `_DOC_PREFIX` | `"query: "` / `"passage: "` | Préfixes e5 (vider pour bge-m3 etc.). |
| `CHROMA_COLLECTION` | `micro_bot_corpus_e5base` | Collection Chroma. |
| `CHROMA_PERSIST_DIR` | `./storage/chroma` | Persistance Chroma. |
| `BM25_INDEX_PATH` | `./storage/bm25.sqlite3` | Index lexical. |

### RAG

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `130` | Découpage (caractères). |
| `RETRIEVAL_TOP_K` | `5` | Chunks finaux envoyés au LLM. |
| `RETRIEVAL_MIN_SCORE` | `0.79` | Seuil cosinus **anglais**. |
| `RETRIEVAL_MIN_SCORE_MULTILINGUAL` | `0.76` | Seuil **français/arabe**. |
| `HYBRID_ENABLED` | `true` | Active BM25 + RRF. |
| `HYBRID_RRF_K` | `60` | Constante RRF. |
| `HYBRID_CANDIDATE_POOL` | `20` | Candidats par canal avant fusion. |
| `HYBRID_BM25_MIN_SCORE` | `1.0` | Seuil de hit lexical « fort » (sur `-bm25`). |
| `HYBRID_REQUIRE_BOTH` | `false` | `false` = OU, `true` = ET strict. |
| `RERANK_ENABLED` | `true` | Cross-encoder actif. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modèle de reranking (~80 Mo). |
| `RERANK_POOL_SIZE` | `20` | Candidats passés au CE. |
| `RERANK_MIN_SCORE` | `-5.0` | Seuil logit (très lâche : réordonne plus qu'il ne filtre). |
| `QUERY_REWRITE_ENABLED` | `false` | Réécriture des follow-ups. |
| `MULTI_QUERY_ENABLED` | `false` | Variantes de requête. |

### Auth & Discord

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `ADMIN_API_KEY` | — | Header `X-Admin-Key`. |
| `MEMBER_TOKEN` | — | Bearer statique membre. |
| `JWT_SECRET_KEY` | `change-me` | Secret de signature (à changer en prod !). |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` (code) / `720` (exemple) | Durée de vie du JWT. |
| `DISCORD_CLIENT_ID` / `_SECRET` / `_REDIRECT_URI` | — | Application OAuth2. |
| `DISCORD_GUILD_ID` | — | Serveur Micro Club. |
| `DISCORD_MEMBER_ROLE_ID` | `""` | Rôle exigé (vide = simple appartenance). |
| `DISCORD_ADMIN_ROLE_IDS` | `""` | CSV de rôles admin. |
| `DISCORD_ADMIN_USER_IDS` | `""` | CSV d'IDs toujours admin (sans bot). |

---

## Scripts d'évaluation & d'exploitation

| Script | Rôle |
| --- | --- |
| `python scripts/ingest_corpus.py [PATH]` | Ingestion CLI récursive du corpus raw. Options `--force`, `--visibility`, `--ephemeral`, `-v`. |
| `python scripts/rebuild_index.py` | **Reconstruit le BM25** depuis Chroma (répare l'index lexical, sans embedding). `--reingest` re-parse/re-embed tout. |
| `python scripts/ping_llm.py` | Smoke-test LLM (1 requête = 1 unité de quota). Affiche provider + tokens. |
| `python scripts/evaluate.py [--hybrid] [--legacy]` | Évaluation retrieval offline sur corpus labellisé (FR/EN/AR, public+members) : Hit@k, MRR, nDCG@5, refus, étanchéité. Calibre les seuils. Zéro LLM. |
| `python scripts/evaluate_access.py` | 50 questions accès+précision (30 membre / 20 non-membre). Compteur de fuites members (doit être 0). |
| `python scripts/evaluate_corpus.py` | Évaluation sur le **vrai index persistant** (comportement réel du bot). |
| `python scripts/test_50_questions.py` | Bout-en-bout 50 questions via `POST /chat` (ou `--quick` retrieval only). Scoring PASS/FAIL/WARN coloré. |
| `python scripts/test_rag_pipeline.py` | Suite manuelle 6 étapes (chunking → grounding → hybrid → rerank → retrieval → pipeline complet). `--quick`, `--only <etape>`. |

---

## Tests

```bash
pytest                          # toute la suite (~100+ tests)
pytest tests/unit -q            # unitaires rapides
pytest tests/integration -q     # Chroma + ingestion (vrai modèle)
pytest tests/security -q        # SSRF + étanchéité visibilité
pytest tests/api -q             # routes admin/documents
pytest --cov=mc_rag             # couverture
```

| Dossier | Couverture |
| --- | --- |
| `tests/unit/` | BM25, chunking, fusion hybride, rewriting, ingestion status, providers LLM, embedder, query logs, parseurs tabulaires. |
| `tests/integration/` | Chroma CRUD + filtres, roundtrip embeddings réels, ingestion complète (idempotence, re-index, ré-embedding incrémental, jobs sérialisés). |
| `tests/security/` | Garde SSRF (17 tests : IPs internes, redirects, metadata cloud), filtrage de visibilité public/members dans les deux canaux. |
| `tests/api/` | Upload 202 + indexation background, ownership, rejets (vide/unsupported/oversized sans résidu), repetitive-questions admin-only. |

Config : `asyncio_mode = "auto"`, `pythonpath = ["src"]` (pyproject.toml).

---

## Développement

```bash
ruff check .                    # lint (line-length 100, règles E,F,I,UP,B)
ruff format .                   # formatage
mypy src                        # type-check strict
uvicorn mc_rag.main:app --reload  # dev server hot-reload
```

Points d'architecture utiles au contributeur :
- Les `Settings` sont cachées (`lru_cache`) — les modifications de `.env` demandent un redémarrage.
- `get_embedder()` / `get_reranker()` sont des singletons processus (chargement paresseux + lock).
- Le warmup d'embedding se fait dans le `lifespan` via un executor : le serveur répond au health pendant le chargement.
- Routeur API : ajouter une route = créer le fichier dans `api/routes/` + l'inclure dans `api/router.py`.
- Les stubs `document_service.py`, `feedback_service.py`, `database.py`, `document_repository.py` sont des marqueurs d'intention — la logique vit dans les services/routes listés plus haut.

---

## Limites connues & roadmap

### Limites actuelles (documentées dans `EDGE_CASES.md`)

- PDF scannés (images) : aucune OCR — erreur `EmptyDocumentError` explicite.
- Membership Discord non révalué pendant la durée du JWT (expiry/revocation non propagée en temps réel).
- State OAuth et jobs d'ingestion **en mémoire par processus** : perdus au redémarrage, non partagés entre workers multi-process.
- Risque DNS rebinding résiduel (fenêtre entre vérification IP et connexion).
- Contenu multilingue mixte dans un même document non géré au chunking.
- `COOKIE_SECURE` / `JWT_ALGORITHM` déclarés mais non appliqués (JWT toujours HS256, pas de cookies).

### Roadmap

- [x] Recherche hybride BM25 + RRF
- [x] Reranking cross-encoder
- [x] Query rewriting + multi-query
- [x] Ingestion incrémentale + progression
- [x] Docker compose + healthcheck
- [ ] Rate limiting sur `/chat` et `/feedback`
- [ ] Refresh planifié des sources URL / Google Sheets
- [ ] Retrieval parent-document pour chunks longs
- [ ] Text-to-SQL sur les données tabulaires indexées
- [ ] Conscience temporelle (dates de documents, pondération de récence)
- [ ] OCR pour PDF scannés

---

## Licence

MIT — voir [LICENSE](LICENSE).

---

<p align="center">
  <img src="docs/logo-icon.jpg" alt="Micro Club" width="64">
</p>

<p align="center">Conçu pour <b>Micro Club (MC)</b> — Université des Sciences et de la Technologie Houari Boumediene.<br>
Design system UI : <a href="https://mcoli-ui.microclub.info/">mcoli-ui</a> (thème Primary, tokens appliqués en HTML/CSS/JS pur).</p>
