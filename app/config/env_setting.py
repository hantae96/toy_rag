import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.environ.get("APP_ENV")
if not APP_ENV:
    raise RuntimeError("APP_ENV must be set (e.g. development or production)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    APP_ENV: str
    OPENAI_API_KEY: str = Field(repr=False)

    MANUAL_DB_URL: str
    RAG_DB_URL: str
    DB_CONTEXT_LIMIT: int
    DB_DOC_MAX_CHARS: int
    EMBEDDING_BATCH_SIZE: int

    PORT: int

    REDIS_DOC_CACHE_KEY: str
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str | None = Field(default=None, repr=False)
    REDIS_SSL: bool
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_SSL: bool = False
    CHROMA_COLLECTION_NAME: str = "manual_chunks"

    IMAGE_BASE_URL: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str = Field(repr=False)
    LANGSMITH_PROJECT: str

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / f".env.{APP_ENV}"),
        env_file_encoding="utf-8",
    )
