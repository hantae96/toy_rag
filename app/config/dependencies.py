from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated

import chromadb
from fastapi import Depends, Request
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaEmbeddings
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.env_setting import Settings
from app.repository.cache_repository import CacheRepository
from app.repository.chat_request_history_repository import ChatRequestHistoryRepository
from app.repository.manual_repository import ManualRepository
from app.repository.preprocess_doc_repository import PreprocessDocRepository
from app.service.chat_service import ChatService
from app.service.document_chunking_service import DocumentChunkingService
from app.service.document_sync_service import DocumentSyncService
from app.service.llm_service import LlmService


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_manual_db_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.manual_db_session_factory


def get_rag_db_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.rag_db_session_factory


async def get_manual_db(
    manual_db_session_factory: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_manual_db_session_factory)
    ],
) -> AsyncGenerator[AsyncSession, None]:
    async with manual_db_session_factory() as db:
        yield db


async def get_rag_db(
    rag_db_session_factory: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_rag_db_session_factory)
    ],
) -> AsyncGenerator[AsyncSession, None]:
    async with rag_db_session_factory() as db:
        yield db


def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis_client


def get_vector_store(request: Request) -> Chroma:
    return request.app.state.vector_store


def get_chat_llm(request: Request) -> Runnable:
    return request.app.state.chat_llm


def get_vision_llm(request: Request) -> Runnable:
    return request.app.state.vision_llm


def create_redis_client(settings: Settings) -> Redis:
    redis_client = Redis(
        host=settings.REDIS_HOST.strip(),
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        ssl=settings.REDIS_SSL,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    redis_client.ping()
    return redis_client


def create_vector_store(settings: Settings) -> Chroma:
    embeddings = OllamaEmbeddings(model="bge-m3")
    chroma_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        ssl=settings.CHROMA_SSL,
    )
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=chroma_client,
    )


def get_manual_repository(
    db: Annotated[AsyncSession, Depends(get_manual_db)],
) -> ManualRepository:
    return ManualRepository(db=db)


def get_chat_request_history_repository(
    db: Annotated[AsyncSession, Depends(get_rag_db)],
) -> ChatRequestHistoryRepository:
    return ChatRequestHistoryRepository(db=db)


def get_preprocess_doc_repository(
    db: Annotated[AsyncSession, Depends(get_rag_db)],
) -> PreprocessDocRepository:
    return PreprocessDocRepository(db=db)


def get_cache_repository(
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CacheRepository:
    return CacheRepository(
        redis_client=redis_client,
        cache_prefix=settings.REDIS_DOC_CACHE_KEY,
        settings=settings,
    )


def get_document_chunking_service() -> DocumentChunkingService:
    return DocumentChunkingService()


def get_llm_service(
    chat_llm: Annotated[Runnable, Depends(get_chat_llm)],
    vision_llm: Annotated[Runnable, Depends(get_vision_llm)],
) -> LlmService:
    return LlmService(chat_llm=chat_llm, vision_llm=vision_llm)


def get_document_sync_service(
    manual_repository: Annotated[ManualRepository, Depends(get_manual_repository)],
    preprocess_doc_repository: Annotated[PreprocessDocRepository, Depends(get_preprocess_doc_repository)],
    document_cache_repository: Annotated[CacheRepository, Depends(get_cache_repository)],
    llm_service: Annotated[LlmService, Depends(get_llm_service)],
    chunking_service: Annotated[DocumentChunkingService, Depends(get_document_chunking_service)],
    vector_store: Annotated[Chroma, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentSyncService:
    return DocumentSyncService(
        manual_repository=manual_repository,
        preprocess_doc_repository=preprocess_doc_repository,
        document_cache_repository=document_cache_repository,
        llm_service=llm_service,
        chunking_service=chunking_service,
        vector_store=vector_store,
        settings=settings,
    )


def get_chat_service(
    document_sync_service: Annotated[DocumentSyncService, Depends(get_document_sync_service)],
    llm_service: Annotated[LlmService, Depends(get_llm_service)],
) -> ChatService:
    return ChatService(
        document_sync_service=document_sync_service,
        llm_service=llm_service,
    )
