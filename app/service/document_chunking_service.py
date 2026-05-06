from langchain_core.documents import Document


class DocumentChunkingService:
    def normalize_text(self, text: str) -> str:
        return "\n".join(
            line.strip() for line in str(text).splitlines() if line.strip()
        ).strip()

    def split_for_embedding(self, text: str) -> list[str]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        return [normalized]

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
