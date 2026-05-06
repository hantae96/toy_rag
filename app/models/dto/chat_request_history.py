from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ChatRequestHistory:
    id: int
    question: str
    answer: str | None
    response_time_ms: int | None
    created_at: datetime
