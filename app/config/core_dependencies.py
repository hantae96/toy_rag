from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.env_setting import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_manual_db_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.manual_db_session_factory


def get_rag_db_session_factory(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
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
