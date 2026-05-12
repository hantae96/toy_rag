import logging
import re
from markdownify import markdownify as md

from bs4 import BeautifulSoup
from bs4.element import Comment
from fastapi.concurrency import run_in_threadpool

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config.env_setting import Settings
from app.models.dto.manual_row import ManualRow
from app.models.dto.sync_doc import Manual
from app.repository.cache_repository import CacheRepository
from app.repository.manual_repository import ManualRepository
from app.repository.preprocess_doc_repository import PreprocessDocRepository
from app.service.document_chunking_service import DocumentChunkingService
from app.service.llm_service import LlmService

logger = logging.getLogger(__name__)

class DocumentSyncService:
    def __init__(
        self,
        manual_repository: ManualRepository,
        preprocess_doc_repository: PreprocessDocRepository,
        document_cache_repository: CacheRepository,
        llm_service: LlmService,
        chunking_service: DocumentChunkingService,
        vector_store: Chroma,
        settings: Settings,
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
        # latest_rows = await self.manual_repository.load_all_manual_rows()

        # # 나중에 여기서 문서 로드 및 파싱 하도록 변경

        # latest_row_by_id = {str(row.id): row for row in latest_rows}
        # latest_doc_ids = set(latest_row_by_id.keys())

        # cached_doc_ids =  self.document_cache_repository.get_doc_ids()

        # # 변경이 있는 문서 ID
        # removed_doc_ids = sorted(cached_doc_ids - latest_doc_ids)
        # new_doc_ids = sorted(latest_doc_ids - cached_doc_ids)

        # if removed_doc_ids:
        #     self._delete_docs_from_vector_store(removed_doc_ids)
        #     self.document_cache_repository.delete_doc_ids(removed_doc_ids)

        # docs_to_add: list[Manual] = []
        # for doc_id in new_doc_ids:
        #     manual = self._prepare_doc_from_row(row=latest_row_by_id[doc_id])
        #     docs_to_add.append(manual)

        # if docs_to_add:
        #     chunk_documents = await self._chunk_docs_for_vector_store(docs_to_add)
            
        #     if chunk_documents:
        #       # 각 청크의 Document.id를 미리 채워두었기 때문에 ids 인자를 따로 넘기지 않는다.
        #       self.vector_store.add_documents(chunk_documents)

        #     # 청크된 문서 캐쉬 갱신
        #     inserted_doc_ids = {str(doc.metadata["doc_id"]) for doc in chunk_documents}
        #     if inserted_doc_ids:
        #       self.document_cache_repository.add_doc_ids(sorted(inserted_doc_ids))

        logger.info("메뉴얼 DB <-> 벡터 DB 동기화 완료")


    def _delete_docs_from_vector_store(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self.vector_store.delete(where={"doc_id": doc_id})

    def _prepare_doc_from_row(self, row: ManualRow) -> Manual:
        content_sequence = self._extract_content_sequence(str(row.content))
        
        
        clean_text = f"제목: {row.title}\n" + f"작성자: {row.writed_by}\n" + "본문:\n" + self._extract_text_from_sequence(content_sequence)
        image_urls = self._extract_image_urls_from_sequence(content_sequence)
        
        return Manual(
            id=str(row.id),
            post_id=str(row.post_id),
            version=str(row.version),
            title=str(row.title),
            doc_url=f"{self.image_base_url}/general/manual/{row.category_id}/view/{row.post_id}",
            text=clean_text,
            image_urls=image_urls,
            content_sequence=content_sequence,
            writed_by=row.writed_by
        )

    async def _chunk_docs_for_vector_store(
        self, docs: list[Manual]
    ) -> list[Document]:
        chunk_documents: list[Document] = []

        for doc in docs:
            # await self.preprocess_doc_repository.upsert_doc(doc.id,doc.text)
            doc_chunks = self.chunking_service.build_chunks(doc.text)
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

    def _extract_content_sequence(self, html: str) -> list[dict[str, str]]:
      if not html:
          return []

      soup = BeautifulSoup(html, "html.parser")

      # script, style, 주석 제거
      for tag in soup(["script", "style"]):
          tag.decompose()
      for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
          comment.extract()

      # 이미지 URL 수집 + img 태그를 플레이스홀더로 교체
      image_urls = []
      for img in soup.find_all("img"):
          src = str(img.attrs.get("src", ""))
          if not src:
              img.decompose()
              continue
          
          url = self._build_image_url(src)
          
          index = len(image_urls)  # append 전에 인덱스 확정
          image_urls.append(url)
          
          img.replace_with(f"{{IMAGE_{index}}}")  # 0, 1, 2... 순서대로

      # HTML → Markdown 변환
      markdown_text = md(str(soup), heading_style="ATX")

      # # 이미지 설명 생성 (있을 경우)
      # if image_urls:
      #     descriptions = self.llm_service.describe_images(image_urls)
      #     for i, desc in enumerate(descriptions):
      #         # 플레이스홀더를 실제 설명으로 교체
      #         markdown_text = markdown_text.replace(
      #             f"{{IMAGE_{i}}}", f"\n[이미지 설명: {desc}]\n"
      #         )

      return [{"type": "text", "value": markdown_text}]

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
