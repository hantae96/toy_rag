from typing import Annotated

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.core_dependencies import get_rag_db
from app.models.dto.chat_request_history import ChatRequestHistory


class ChatRequestHistoryRepository:
    def __init__(
        self,
        db: Annotated[AsyncSession, Depends(get_rag_db)],
    ) -> None:
        self.db = db

    async def ensure_table(self) -> None:
        create_table_query = text(
            """
            CREATE TABLE IF NOT EXISTS chat_request_histories (
                id BIGSERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT,
                response_time_ms INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        add_answer_column_query = text(
            """
            ALTER TABLE chat_request_histories
            ADD COLUMN IF NOT EXISTS answer TEXT
            """
        )
        add_response_time_column_query = text(
            """
            ALTER TABLE chat_request_histories
            ADD COLUMN IF NOT EXISTS response_time_ms INTEGER
            """
        )
        await self.db.execute(create_table_query)
        await self.db.execute(add_answer_column_query)
        await self.db.execute(add_response_time_column_query)
        await self.db.commit()

    async def save_chat_request_history(
        self,
        question: str,
        answer: str,
        response_time_ms: int,
    ) -> int:
        query = text(
            """
            INSERT INTO chat_request_histories (question, answer, response_time_ms)
            VALUES (:question, :answer, :response_time_ms)
            RETURNING id
            """
        )
        note_id = (
            await self.db.execute(
                query,
                {
                    "question": question,
                    "answer": answer,
                    "response_time_ms": response_time_ms,
                },
            )
        ).scalar_one()
        await self.db.commit()
        return int(note_id)

    async def list_recent(self, limit: int = 20) -> list[ChatRequestHistory]:
        query = text(
            """
            SELECT id, question, answer, response_time_ms, created_at
            FROM chat_request_histories
            ORDER BY id DESC
            LIMIT :limit
            """
        )
        rows = (
            await self.db.execute(
                query,
                {"limit": limit},
            )
        ).mappings().all()

        return [
            ChatRequestHistory(
                id=int(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]) if row["answer"] is not None else None,
                response_time_ms=(
                    int(row["response_time_ms"])
                    if row["response_time_ms"] is not None
                    else None
                ),
                created_at=row["created_at"],
            )
            for row in rows
        ]
