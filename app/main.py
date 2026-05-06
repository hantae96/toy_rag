from contextlib import asynccontextmanager
import logging

import chromadb
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.env_setting import Settings
from app.core.database import (
    create_manual_db_engine,
    create_rag_db_engine,
    create_session_factory,
)
from app.models.api.image_models import ImageDescriptions
from app.repository.cache_repository import CacheRepository
from app.repository.chat_request_history_repository import ChatRequestHistoryRepository
from app.repository.manual_repository import ManualRepository
from app.repository.preprocess_doc_repository import PreprocessDocRepository
from app.router.chat_router import router as chat_router
from app.service.document_chunking_service import DocumentChunkingService
from app.service.document_sync_service import DocumentSyncService
from app.service.llm_service import LlmService

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s - %(asctime)s ",
)
logging.getLogger().setLevel(logging.INFO)


def _create_redis_client(settings: Settings) -> Redis:
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


def _create_vector_store(settings: Settings) -> Chroma:
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


async def _run_startup_sync(
    settings: Settings,
    manual_db_session_factory: async_sessionmaker[AsyncSession],
    rag_db_session_factory: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    vector_store: Chroma,
    chat_llm: ChatOpenAI,
    vision_llm,
) -> None:
    async with manual_db_session_factory() as manual_db, rag_db_session_factory() as rag_db:
        manual_repository = ManualRepository(db=manual_db)
        preprocess_doc_repository = PreprocessDocRepository(db=rag_db)
        chat_request_history_repository = ChatRequestHistoryRepository(db=rag_db)

        await preprocess_doc_repository.ensure_table()
        await chat_request_history_repository.ensure_table()

        document_sync_service = DocumentSyncService(
            manual_repository=manual_repository,
            preprocess_doc_repository=preprocess_doc_repository,
            document_cache_repository=CacheRepository(
                redis_client=redis_client,
                settings=settings,
                cache_prefix=settings.REDIS_DOC_CACHE_KEY,
            ),
            llm_service=LlmService(
                chat_llm=chat_llm,
                vision_llm=vision_llm,
            ),
            chunking_service=DocumentChunkingService(),
            vector_store=vector_store,
            settings=settings,
        )
        await document_sync_service.load_documents()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()  # type: ignore[call-arg] Setting은 런타임에서 처리

    manual_db_engine = create_manual_db_engine(settings=settings)
    manual_db_session_factory = create_session_factory(manual_db_engine)

    rag_db_engine = create_rag_db_engine(settings=settings)
    rag_db_session_factory = create_session_factory(rag_db_engine)

    redis_client = _create_redis_client(settings=settings)
    vector_store = _create_vector_store(settings=settings)
    chat_llm = ChatOpenAI(model="gpt-4o-mini")
    vision_llm = ChatOpenAI(model="gpt-4o").with_structured_output(
        ImageDescriptions
    )

    await _run_startup_sync(
        settings=settings,
        manual_db_session_factory=manual_db_session_factory,
        rag_db_session_factory=rag_db_session_factory,
        redis_client=redis_client,
        vector_store=vector_store,
        chat_llm=chat_llm,
        vision_llm=vision_llm,
    )

    app.state.settings = settings
    app.state.manual_db_engine = manual_db_engine
    app.state.manual_db_session_factory = manual_db_session_factory
    app.state.rag_db_engine = rag_db_engine
    app.state.rag_db_session_factory = rag_db_session_factory
    app.state.redis_client = redis_client
    app.state.vector_store = vector_store
    app.state.chat_llm = chat_llm
    app.state.vision_llm = vision_llm

    yield

    redis_client.close()
    await rag_db_engine.dispose()
    await manual_db_engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router)
