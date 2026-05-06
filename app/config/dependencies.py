from typing import Annotated

from fastapi import Depends

from app.repository.chat_request_history_repository import ChatRequestHistoryRepository
from app.service.chat_service import ChatService


def get_chat_service(
    service: Annotated[ChatService, Depends()],
) -> ChatService:
    return service


def get_chat_request_history_repository(
    repository: Annotated[ChatRequestHistoryRepository, Depends()],
) -> ChatRequestHistoryRepository:
    return repository
