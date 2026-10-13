"""Centralized application settings loaded from the environment.

Settings are read from real environment variables first and fall back to a local
``.env`` file (see ``.env.example``). Import :func:`get_settings` rather than
instantiating :class:`Settings` directly so the parsed configuration is cached.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
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
    gemini_api_key: str | None = None
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
    discord_redirect_uri: str = "http://localhost:8000/api/auth/discord/callback"
    discord_guild_id: str | None = None
    discord_bot_token: str | None = None

    # --- Auth / security ---------------------------------------------------
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    member_token: str | None = None
    admin_api_key: str | None = None
    conversation_db_path: str = "./storage/conversations.db"
    feedback_db_path: str = "./storage/feedback.db"
    upload_dir: str = "./data/uploads"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

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