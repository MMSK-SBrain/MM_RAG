"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration is loaded from .env or environment variables."""

    # --- LLM / OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Embedding model (local via HuggingFace) ---
    embedding_model_name: str = "ibm-granite/granite-embedding-30m-english"

    # --- Granite models on OpenRouter ---
    granite_instruct_model: str = "ibm-granite/granite-3.1-8b-instruct"
    granite_vision_model: str = "ibm-granite/granite-vision-3.1-2b"

    # --- FAISS ---
    faiss_index_path: str = "faiss_index"

    # --- RAG parameters ---
    retriever_top_k: int = 4
    max_query_length: int = 1000

    # --- Rate limiting ---
    rate_limit_default: str = "30/minute"
    rate_limit_ingest: str = "5/minute"

    # --- Mock mode (for demos without API keys) ---
    mock_mode: bool = False

    # --- Upload settings ---
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
