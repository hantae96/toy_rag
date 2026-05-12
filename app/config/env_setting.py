import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.environ.get("APP_ENV")
if not APP_ENV:
    raise RuntimeError("APP_ENV must be set (e.g. development or production)")

_env_file = Path(__file__).resolve().parents[2] / f".env.{APP_ENV}"
load_dotenv(_env_file)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_ENV: str
    PORT: int

    # OpenAI
    OPENAI_API_KEY: str
    CHAT_LLM_MODEL: str = "gpt-4o-mini"
    VISION_LLM_MODEL: str = "gpt-4o"

    # DB
    MANUAL_DB_URL: str
    RAG_DB_URL: str
    DB_CONTEXT_LIMIT: int
    DB_DOC_MAX_CHARS: int
    EMBEDDING_BATCH_SIZE: int
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str | None = None
    REDIS_SSL: bool = False
    REDIS_DOC_CACHE_KEY: str

    # Chroma
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_SSL: bool = False
    CHROMA_COLLECTION_NAME: str = "manual_chunks"
    EMBEDDING_PROVIDER: str = "ollama"  # ollama | openai
    EMBEDDING_MODEL: str = "bge-m3"

    # Image
    IMAGE_BASE_URL: str

    # LangSmith
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
