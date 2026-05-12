from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dto.preprocess_doc import PreprocessDoc


class PreprocessDocRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def execute(self, query, params: dict | None = None, commit: bool = False):
        result = await self.db.execute(query, params)
        if commit:
            await self.db.commit()
        return result

    async def ensure_table(self) -> None:
        query = text(
            """
            CREATE TABLE IF NOT EXISTS preprocess_docs (
                doc_id TEXT PRIMARY KEY,
                preprcess_text TEXT NOT NULL
            )
            """
        )
        await self.execute(query, commit=True)

    async def upsert_doc(
        self,
        doc_id: str,
        preprcess_text: str,
    ) -> str:
        query = text(
            """
            INSERT INTO preprocess_docs (doc_id, preprcess_text)
            VALUES (:doc_id, :preprcess_text)
            ON CONFLICT (doc_id)
            DO UPDATE SET
                preprcess_text = EXCLUDED.preprcess_text
            RETURNING doc_id
            """
        )
        saved_doc_id = (
            await self.execute(
                query,
                {
                    "doc_id": doc_id,
                    "preprcess_text": preprcess_text,
                },
                commit=True,
            )
        ).scalar_one()
        return str(saved_doc_id)

    async def get_doc_by_id(self, doc_id: str) -> PreprocessDoc | None:
        query = text(
            """
            SELECT doc_id, preprcess_text
            FROM preprocess_docs
            WHERE doc_id = :doc_id
            """
        )
        row = (
            await self.execute(
                query,
                {"doc_id": doc_id},
            )
        ).mappings().first()

        if row is None:
            return None

        return PreprocessDoc(
            doc_id=str(row["doc_id"]),
            preprcess_text=str(row["preprcess_text"]),
        )

    async def list_docs(self, limit: int = 100) -> list[PreprocessDoc]:
        query = text(
            """
            SELECT doc_id, preprcess_text
            FROM preprocess_docs
            ORDER BY doc_id
            LIMIT :limit
            """
        )
        rows = (
            await self.execute(
                query,
                {"limit": limit},
            )
        ).mappings().all()

        return [
            PreprocessDoc(
                doc_id=str(row["doc_id"]),
                preprcess_text=str(row["preprcess_text"]),
            )
            for row in rows
        ]

    async def delete_doc_by_id(self, doc_id: str) -> int:
        query = text(
            """
            WITH deleted AS (
                DELETE FROM preprocess_docs
                WHERE doc_id = :doc_id
                RETURNING 1
            )
            SELECT COUNT(*) AS deleted_count FROM deleted
            """
        )
        result = await self.execute(
            query,
            {"doc_id": doc_id},
            commit=True,
        )
        return int(result.scalar_one())
