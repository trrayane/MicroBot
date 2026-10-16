"""Centralized application settings loaded from the environment.

Settings are read from real environment variables first and fall back to a local
``.env`` file (see ``.env.example``). Import :func:`get_settings` rather than
instantiating :class:`Settings` directly so the parsed configuration is cached.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the mc-club-rag service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "mc-club-rag"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- LLM (OpenRouter) --------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "mc-club-rag"
    # Native Google Gemini (only used when GEMINI_API_KEY is set).
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = "gemini-3.5-flash-lite"

    # --- Embeddings (local) ------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    embedding_device: str | None = None
    embedding_normalize: bool = True
    # e5 models require these prefixes. Set both to "" for models without them
    # (e.g. bge-m3).
    embedding_query_prefix: str = "query: "
    embedding_doc_prefix: str = "passage: "

    # --- Vector store (Chroma) ---------------------------------------------
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection: str = "micro_bot_corpus_e5base"
    chroma_persist: bool = True

    # --- Database ----------------------------------------------------------
    database_url: str = "sqlite:///./storage/mc_club.db"

    # --- Discord OAuth / membership ----------------------------------------
    discord_client_id: str | None = None
    discord_client_secret: str | None = None
    discord_redirect_uri: str = "http://localhost:8000/api/v1/auth/discord/callback"
    discord_guild_id: str | None = None
    discord_bot_token: str | None = None
    # Role required to be considered a member ("" = just being in the guild).
    discord_member_role_id: str = ""
    # Comma-separated Discord role IDs that grant admin access.
    discord_admin_role_ids: str = ""
    # Comma-separated Discord user IDs that are always admin (no bot needed).
    discord_admin_user_ids: str = ""

    # --- Auth / security ---------------------------------------------------
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    member_token: str | None = None
    admin_api_key: str | None = None
    conversation_db_path: str = "./storage/conversations.db"
    feedback_db_path: str = "./storage/feedback.db"
    users_db_path: str = "./storage/users.db"
    upload_dir: str = "./data/uploads"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    frontend_url: str = "http://localhost:8000"
    cookie_secure: bool = False

    @property
    def admin_role_ids(self) -> frozenset[str]:
        """Discord role IDs that grant admin access."""
        return frozenset(
            r.strip() for r in self.discord_admin_role_ids.split(",") if r.strip()
        )

    @property
    def admin_user_ids(self) -> frozenset[str]:
        """Discord user IDs that are always admin (bypasses guild/role checks)."""
        return frozenset(
            u.strip() for u in self.discord_admin_user_ids.split(",") if u.strip()
        )

    # --- Ingestion / RAG tuning --------------------------------------------
    raw_data_dir: str = "./data/raw"
    chunk_size: int = 900
    chunk_overlap: int = 130
    retrieval_top_k: int = 5
    # e5 cosine scores are compressed (roughly 0.7-0.9). Calibrated with
    # scripts/evaluate.py and scripts/evaluate_corpus.py (see scripts/).
    #
    # English: in-scope >= 0.792, out-of-scope <= 0.770 across the labeled and
    # live corpora, so 0.79 sits in the gap.
    retrieval_min_score: float = 0.79
    # Cross-lingual (French/Arabic) against English sources scores lower, so it
    # gets its own threshold. Combined constraint over both corpora: in-scope
    # (live) >= 0.761, out-of-scope (labeled) <= 0.755 -> 0.76 separates them.
    retrieval_min_score_multilingual: float = 0.76

    # --- Spreadsheets (Excel / CSV / Google Sheets) ------------------------
    # One chunk per row; rows are never cut in the middle.
    sheet_max_rows: int = 50000
    sheet_max_cell_chars: int = 500
    # Short rows are grouped together until they reach this many characters.
    sheet_min_row_chars: int = 200
    sheet_max_rows_per_chunk: int = 8

    # --- URL fetching (SSRF guard) -----------------------------------------
    url_fetch_timeout: float = 10.0
    url_fetch_max_redirects: int = 5

    # --- Hybrid retrieval (BM25 + vector, fused with RRF) ------------------
    # On by default: BM25 catches exact terms the embedder might miss, and RRF
    # fuses both rankings without needing to reconcile cosine and BM25 scales.
    hybrid_enabled: bool = True
    hybrid_rrf_k: int = 60
    # Candidates fetched per channel before fusing and cutting to top_k.
    hybrid_candidate_pool: int = 20
    # "Strong" BM25 hit: -bm25(raw) must reach this value. FTSS5's bm25() is
    # negative (lower = better) and corpus-dependent: a term present in most
    # chunks scores ~0, a rare term scores several units.
    hybrid_bm25_min_score: float = 1.0
    # True -> answer only when BOTH channels pass; False -> either one passes.
    hybrid_require_both: bool = False
    bm25_index_path: str = "./storage/bm25.sqlite3"

    # --- Cross-encoder reranking -------------------------------------------
    # Reranks the fused candidate pool with a cross-encoder for precision.
    # MiniLM ~80 MB / ~2 s load. For stronger multilingual precision swap to
    # BAAI/bge-reranker-v2-m3 (~2 GB, ~30-60 s first load).
    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Candidates fed to the cross-encoder (before cutting to top_k).
    rerank_pool_size: int = 20
    # Cross-encoder logit threshold; below this the chunk is dropped.
    rerank_min_score: float = -5.0

    # --- Query rewriting / multi-query (conditional, at most ONE LLM call) --
    query_rewrite_enabled: bool = False
    # Below this first-pass cosine score a rewrite is attempted. 0 = use the
    # language threshold (RETRIEVAL_MIN_SCORE / _MULTILINGUAL).
    query_rewrite_min_score: float = 0.0
    query_rewrite_max_chars: int = 300
    multi_query_enabled: bool = False
    multi_query_count: int = 2
    multi_query_max_chars: int = 200

    @field_validator(
        "openrouter_api_key",
        "openrouter_site_url",
        "gemini_api_key",
        "embedding_device",
        "discord_client_id",
        "discord_client_secret",
        "discord_guild_id",
        "discord_bot_token",
        "member_token",
        "admin_api_key",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat an empty/whitespace env value (e.g. ``EMBEDDING_DEVICE=``) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """Return ``cors_origins`` split into individual origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()