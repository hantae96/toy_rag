from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dto.manual_row import ManualRow


class ManualRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def count_manuals(self) -> int:
        query = text(f"SELECT COUNT(*) AS total FROM holdings_manual.post_history")
        total = (await self.db.execute(query)).scalar_one()
        return int(total)
    
    async def get_manual_count_by_user_id(self, user_id: str) -> int:
        query = text(
            """
                SELECT count(*) FROM post_history AS a
                JOIN `user` AS b ON a.user_id = b.id
                WHERE b.name LIKE :pattern
            """
        )
        total = (await self.db.execute(query, {"pattern": f"%{user_id}%"})).scalar_one()
        return int(total)

    # TODO: 임시로 595 doc_id 적용 테스트 후 제거
    async def load_all_manual_rows(self) -> list[ManualRow]:
        query = text(
            f"""
            SELECT a.id,a.post_id,a.category_id,a.category,a.version,a.title,a.content,c.name 
            FROM (
                SELECT a.id,a.post_id,a.category as category_id, b.name AS category, a.version,a. title,a.content, a.user_id,
                    ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY version DESC) as rn
                FROM holdings_manual.post_history AS a
                 JOIN holdings_manual.menu_item AS b ON a.category = b.id  
            ) AS a -- 가장 최신 문서 정보 호출
            JOIN post_approval AS b ON a.id = b.post_history_id
            JOIN `user` AS c ON a.user_id = c.id
            WHERE a.rn = 1
              AND a.id = 595
            	AND b.approval = 2 -- 최종 승인이 난 문서만 호출
            ORDER BY a.id
            """
        )
        rows = (await self.db.execute(query)).mappings().all()

        return [
            ManualRow(
                id=row["id"],
                post_id=row["post_id"],
                version=row["version"],
                title=row["title"],
                content=row["content"],
                category=row["category"],
                category_id=row["category_id"],
                writed_by= row["name"]
            )
            for row in rows
        ]
