from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.env_setting import Settings

def create_manual_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.MANUAL_DB_URL, pool_pre_ping=True)


def create_rag_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.RAG_DB_URL, pool_pre_ping=True)


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, expire_on_commit=False)
