import logging
from typing import Annotated
from typing import cast

from fastapi import Depends
from redis import Redis

from app.config.core_dependencies import get_redis_client, get_settings
from app.config.env_setting import Settings

logger = logging.getLogger(__name__)


class CacheRepository:
    def __init__(
        self,
        redis_client: Annotated[Redis, Depends(get_redis_client)],
        settings: Annotated[Settings, Depends(get_settings)],
        cache_prefix: str | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.cache_prefix = cache_prefix or settings.REDIS_DOC_CACHE_KEY
        self.index_key = f"{self.cache_prefix}:index"

    def get_doc_ids(self) -> set[str]:
        try:
            members_result = self.redis_client.smembers(self.index_key)
            members = cast(set[str], members_result)
            return {str(member).strip() for member in members if str(member).strip()}

        except Exception as e:
            logger.info(f"Redis 인덱스 조회 실패: {e}")
            return set()

    def add_doc_ids(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return

        try:
            self.redis_client.sadd(self.index_key, *doc_ids)

            logger.info(
                "Redis 문서 인덱스 추가 완료 (%s건)",
                len(doc_ids),
            )
        except Exception as e:
            logger.info(f"Redis 인덱스 추가 실패: {e}")

    def delete_doc_ids(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return

        try:
            self.redis_client.srem(self.index_key, *doc_ids)

            logger.info(
                "Redis 문서 인덱스 삭제 완료 (%s건)",
                len(doc_ids),
            )
        except Exception as e:
            logger.info(f"Redis 인덱스 삭제 실패: {e}")
