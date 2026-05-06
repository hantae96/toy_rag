import logging
import time

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from fastapi.concurrency import run_in_threadpool

from app.models.api.chat_models import ApiResponse
from app.service.document_sync_service import DocumentSyncService
from app.service.llm_service import LlmService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        document_sync_service: Annotated[DocumentSyncService, Depends()],
        llm_service: Annotated[LlmService, Depends()],
    ) -> None:
        self.document_sync_service = document_sync_service
        self.llm_service = llm_service

    async def load_documents(self) -> None:
        await self.document_sync_service.load_documents()

    async def ask(self, question: str) -> ApiResponse:
        context = await self.document_sync_service.build_context(question)
        return await run_in_threadpool(
            self.llm_service.ask,
            question,
            context,
        )

    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        
        context_start = time.perf_counter()

        context = await self.document_sync_service.build_context(question)

        context_end = time.perf_counter()
        logger.info(f"context 생성 시간 : {(context_end - context_start) * 1000:.2f} ms")

        llm_start = time.perf_counter()
        async for chunck in self.llm_service.ask_stream(question=question, context=context):
            yield chunck
        llm_end = time.perf_counter()
        logger.info(f"llm 응답 생성 시간 : {(llm_end - llm_start) * 1000:.2f} ms")
        
