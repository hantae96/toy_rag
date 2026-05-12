from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re


class DocumentChunkingService:
    def __init__(self):
        # 일반 텍스트 청킹용
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def normalize_text(self, text: str) -> str:
        return "\n".join(
            line.strip() for line in str(text).splitlines() if line.strip()
        ).strip()

    def _extract_tables(self, text: str) -> tuple[list[str], str]:
        """표를 추출하고, 표가 제거된 텍스트를 반환"""
        # 마크다운 표 패턴 (헤더 + 구분선 + 데이터행)
        table_pattern = re.compile(
            r"(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)",
            re.MULTILINE
        )

        tables = table_pattern.findall(text)                  # 표 추출
        remaining_text = table_pattern.sub("", text).strip()  # 표 제거된 텍스트

        return tables, remaining_text

    def split_for_embedding(self, text: str) -> list[str]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []

        # 표 추출 + 나머지 텍스트 분리
        tables, remaining_text = self._extract_tables(normalized)

        chunks: list[str] = []

        # 표는 하나의 청크로
        for table in tables:
            if table.strip():
                chunks.append(table.strip())

        # 나머지 텍스트는 일반 청킹
        if remaining_text:
            text_chunks = self.text_splitter.split_text(remaining_text)
            chunks.extend(text_chunks)

        return chunks

    def build_chunks(self, text: str) -> list[Document]:
        split_chunks = self.split_for_embedding(text)
        if not split_chunks:
            return []

        chunks: list[Document] = []
        for chunk_index, chunk_text in enumerate(split_chunks):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={"index": chunk_index},
                )
            )
        return chunks