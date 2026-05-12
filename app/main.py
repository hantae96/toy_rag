from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.dependencies import create_redis_client, create_vector_store, get_settings
from app.core.database import get_manual_db_engine, get_rag_db_engine, get_session_factory
from app.config.logging_config import LoggingConfigurator
from app.middleware.global_exception_middleware import GlobalExceptionMiddleware
from app.middleware.request_log_middleware import RequestIdMiddleware, RequestLogMiddleware
from app.models.api.image_models import ImageDescriptions
from app.repository.chat_request_history_repository import ChatRequestHistoryRepository
from app.repository.preprocess_doc_repository import PreprocessDocRepository
from app.router.chat_router import router as chat_router
from app.router.sync_router import router as sync_router

LoggingConfigurator.configure()


async def _ensure_tables(
    rag_db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with rag_db_session_factory() as rag_db:
        await PreprocessDocRepository(db=rag_db).ensure_table()
        await ChatRequestHistoryRepository(db=rag_db).ensure_table()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    manual_db_engine = get_manual_db_engine(settings=settings)
    manual_db_session_factory = get_session_factory(manual_db_engine)

    rag_db_engine = get_rag_db_engine(settings=settings)
    rag_db_session_factory = get_session_factory(rag_db_engine)

    redis_client = create_redis_client(settings=settings)
    vector_store = create_vector_store(settings=settings)

    api_key = SecretStr(settings.OPENAI_API_KEY)
    chat_llm = ChatOpenAI(model=settings.CHAT_LLM_MODEL, api_key=api_key)
    vision_llm = ChatOpenAI(
        model=settings.VISION_LLM_MODEL, api_key=api_key
    ).with_structured_output(ImageDescriptions)

    await _ensure_tables(rag_db_session_factory=rag_db_session_factory)

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

app.add_middleware(GlobalExceptionMiddleware)
app.add_middleware(RequestLogMiddleware)
app.add_middleware(RequestIdMiddleware)

app.include_router(chat_router)
app.include_router(sync_router)
