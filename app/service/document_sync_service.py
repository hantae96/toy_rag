import logging
from typing import Annotated

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from fastapi import Depends
from fastapi.concurrency import run_in_threadpool

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config.core_dependencies import get_settings, get_vector_store
from app.config.env_setting import Settings
from app.models.dto.manual_row import ManualRow
from app.models.dto.sync_doc import Manual
from app.repository.cache_repository import (
    CacheRepository,
)
from app.repository.manual_repository import ManualRepository
from app.repository.preprocess_doc_repository import PreprocessDocRepository
from app.service.document_chunking_service import DocumentChunkingService
from app.service.llm_service import LlmService

logger = logging.getLogger(__name__)

class DocumentSyncService:
    def __init__(
        self,
        manual_repository: Annotated[ManualRepository, Depends()],
        preprocess_doc_repository: Annotated[PreprocessDocRepository, Depends()],
        document_cache_repository: Annotated[CacheRepository, Depends()],
        llm_service: Annotated[LlmService, Depends()],
        chunking_service: Annotated[DocumentChunkingService, Depends()],
        vector_store: Annotated[Chroma, Depends(get_vector_store)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        
        self.document_cache_repository = document_cache_repository
        self.manual_repository = manual_repository 
        self.preprocess_doc_repository = preprocess_doc_repository

        self.llm_service = llm_service
        self.chunking_service = chunking_service

        self.vector_store = vector_store
        self.image_base_url = settings.IMAGE_BASE_URL

        self.max_context_chunks = 3
        self.max_chunks_per_doc = 2

    async def load_documents(self) -> None:
        latest_rows = await self.manual_repository.load_all_manual_rows()

        # 나중에 여기서 문서 로드 및 파싱 하도록 변경

        latest_row_by_id = {str(row.id): row for row in latest_rows}
        latest_doc_ids = set(latest_row_by_id.keys())

        cached_doc_ids = await run_in_threadpool(
            self.document_cache_repository.get_doc_ids
        )

        removed_doc_ids = sorted(cached_doc_ids - latest_doc_ids)
        new_doc_ids = sorted(latest_doc_ids - cached_doc_ids)

        if removed_doc_ids:
            await run_in_threadpool(self._delete_docs_from_vector_store, removed_doc_ids)
            await run_in_threadpool(
                self.document_cache_repository.delete_doc_ids,
                removed_doc_ids,
            )

        docs_to_add: list[Manual] = []
        for doc_id in new_doc_ids:
            manual = self._prepare_doc_from_row(row=latest_row_by_id[doc_id])
            docs_to_add.append(manual)

        if docs_to_add:
            chunk_documents = await self._chunk_docs_for_vector_store(docs_to_add)
            
            if chunk_documents:
                # 각 청크의 Document.id를 미리 채워두었기 때문에 ids 인자를 따로 넘기지 않는다.
                await run_in_threadpool(
                    self.vector_store.add_documents,
                    chunk_documents,
                )

            # 청크된 문서 캐쉬 갱신
            inserted_doc_ids = {str(doc.metadata["doc_id"]) for doc in chunk_documents}
            if inserted_doc_ids:
                await run_in_threadpool(
                    self.document_cache_repository.add_doc_ids,
                    sorted(inserted_doc_ids),
                )

        logger.info("메뉴얼 DB <-> 벡터 DB 동기화 완료")


    def _delete_docs_from_vector_store(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self.vector_store.delete(where={"doc_id": doc_id})

    def _prepare_doc_from_row(self, row: ManualRow) -> Manual:
        content_sequence = self._extract_content_sequence(str(row.content))

        clean_text = self._extract_text_from_sequence(content_sequence)
        image_urls = self._extract_image_urls_from_sequence(content_sequence)
        
        return Manual(
            id=str(row.id),
            post_id=str(row.post_id),
            version=str(row.version),
            title=str(row.title),
            doc_url=f"{self.image_base_url}/general/manual/{row.category_id}/view/{row.post_id}",
            text=clean_text,
            image_urls=image_urls,
            image_descriptions=[],
            content_sequence=content_sequence,
        )

    async def _chunk_docs_for_vector_store(
        self, docs: list[Manual]
    ) -> list[Document]:
        chunk_documents: list[Document] = []

        for doc in docs:
            # 텍스트 + 이미지 포함 설명
            rendered = self._render_sequence_for_embedding_with_descriptions(
                sequence=doc.content_sequence,
                image_descriptions=doc.image_descriptions,
            )

            await self.preprocess_doc_repository.upsert_doc(doc.id, rendered)
            
            doc_chunks = self.chunking_service.build_chunks(rendered)
            if not doc_chunks:
                continue

            doc_id = str(doc.id)
            for chunk in doc_chunks:
                chunk_index = int(chunk.metadata["index"])
                # 안정적인 upsert/delete를 위해 청크 ID를 Document.id에 고정한다.
                chunk.id = self._chunk_id(doc_id, chunk_index)
                chunk.metadata = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "title": doc.title,
                    "doc_url": doc.doc_url,
                    "post_id": doc.post_id,
                    "version": doc.version,
                }

                chunk_documents.append(chunk)

        return chunk_documents

    def _chunk_id(self, doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}:{chunk_index}"

    async def build_context(self, question: str) -> str:
        docs_and_scores: list[tuple[Document, float]] = await run_in_threadpool(
            self.vector_store.similarity_search_with_score,
            query=question,
            k=self.max_context_chunks,
        )

        context_parts: list[str] = []

        for chunk_doc, _score in docs_and_scores:
            metadata = chunk_doc.metadata

            combined = (
                f"================================\n"
                f"doc_id : {metadata['doc_id']}\n"
                f"title: {metadata['title']}\n"
                f"doc_url :{metadata['doc_url']}\n"
                f"================================\n"
                f"content: {chunk_doc.page_content}\n"
            )

            context_parts.append(combined)

        return "\n\n".join(context_parts)

    def _ensure_image_descriptions(self, docs: list[Manual]) -> None:
        for doc in docs:
            image_urls = doc.image_urls
            existing_descriptions = doc.image_descriptions
            if len(existing_descriptions) == len(image_urls):
                continue

            doc.image_descriptions = self.llm_service.describe_images(image_urls=image_urls)

    def _extract_content_sequence(self, html: str) -> list[dict[str, str]]:
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        sequence: list[dict[str, str]] = []
        for node in soup.descendants:
            if isinstance(node, Tag) and node.name == "img":
                if "src" not in node.attrs:
                    continue
                src = str(node["src"]).strip()
                if not src:
                    continue
                sequence.append(
                    {"type": "image", "value": self._build_image_url(src)}
                )
                continue

            if isinstance(node, NavigableString):
                text = " ".join(str(node).split())
                if text:
                    sequence.append({"type": "text", "value": text})

        return sequence

    def _extract_text_from_sequence(self, sequence: list[dict[str, str]]) -> str:
        text_parts: list[str] = []
        for block in sequence:
            if block["type"] == "text":
                value = str(block["value"]).strip()
                if value:
                    text_parts.append(value)
        return " ".join(text_parts)

    def _extract_image_urls_from_sequence(
        self, sequence: list[dict[str, str]]
    ) -> list[str]:
        image_urls: list[str] = []
        for block in sequence:
            if block["type"] == "image":
                value = str(block["value"]).strip()
                if value:
                    image_urls.append(value)
        return image_urls

    def _render_sequence_for_embedding_with_descriptions(
        self, sequence: list[dict[str, str]], image_descriptions: list[str]
    ) -> str:
        parts: list[str] = []
        image_index = 0

        for block in sequence:
            block_type = block["type"]
            value = str(block["value"]).strip()
            if not value:
                continue

            if block_type == "text":
                parts.append(value)
            elif block_type == "image":
                if image_index < len(image_descriptions):
                    description = str(image_descriptions[image_index]).strip()
                    if description:
                        parts.append(f"[이미지 설명] {description}")
                    else:
                        parts.append("[이미지]")
                else:
                    parts.append("[이미지]")
                image_index += 1

        return "\n".join(parts)

    def _build_image_url(self, src: str) -> str:
        raw_src = str(src).strip()
        if not raw_src:
            return ""

        lower_src = raw_src.lower()
        if lower_src.startswith("http://") or lower_src.startswith("https://"):
            return raw_src

        return str(self.image_base_url) + raw_src

    def _extract_text_from_html(self, html: str) -> str:
        sequence = self._extract_content_sequence(html)
        return self._extract_text_from_sequence(sequence)

    def _extract_image_urls(self, html: str) -> list[str]:
        sequence = self._extract_content_sequence(html)
        return self._extract_image_urls_from_sequence(sequence)
