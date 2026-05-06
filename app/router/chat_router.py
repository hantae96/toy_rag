import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.config.dependencies import (
    get_chat_request_history_repository,
    get_chat_service,
)
from app.models.api.chat_models import ApiResponse, ChatRequest
from app.repository.chat_request_history_repository import ChatRequestHistoryRepository
from app.service.chat_service import ChatService
from langsmith import traceable

router = APIRouter()
logger = logging.getLogger(__name__)


def _to_sse_data(payload: str) -> str:
    lines = payload.splitlines() or [payload]
    return "\n".join(f"data: {line}" for line in lines) + "\n\n"


async def _sse_stream(question: str, service: ChatService) -> AsyncGenerator[str, None]:
    try:
        yield "event: start\ndata: stream-started\n\n"
        async for chunk in service.ask_stream(question):
            data = json.dumps({"chunk": chunk}, ensure_ascii=False)
            yield _to_sse_data(data)
        yield "event: done\ndata: [DONE]\n\n"
    except Exception as e:
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"event: error\n{_to_sse_data(error_data)}"


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ApiResponse)
@traceable
async def chat(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    chat_request_history_repository: ChatRequestHistoryRepository = Depends(
        get_chat_request_history_repository
    ),
) -> ApiResponse:
    answer_start = time.perf_counter()
    response = await service.ask(req.question)
    answer_end = time.perf_counter()
    response_time_ms = int((answer_end - answer_start) * 1000)

    try:
        await chat_request_history_repository.save_chat_request_history(
            question=req.question,
            answer=response.answer,
            response_time_ms=response_time_ms,
        )
    except Exception:
        logger.exception("질문/응답 저장 실패(question=%s)", req.question)

    return response

@router.get("/chat/stream")
@traceable
async def streaming_sse(
    question: str = Query(..., min_length=1),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    logger.info(f"question : {question}")
    return StreamingResponse(
        _sse_stream(question, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
