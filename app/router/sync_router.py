import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.dependencies import get_document_sync_service
from app.service.document_sync_service import DocumentSyncService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sync")
async def sync_documents(
    service: Annotated[DocumentSyncService, Depends(get_document_sync_service)],
) -> dict[str, str]:
    await service.load_documents()
    return {"status": "ok"}
