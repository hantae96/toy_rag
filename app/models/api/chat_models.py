from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")


class ApiResponse(BaseModel):
    answer: str = Field(description="Final answer for API client")
    reference_urls: list[str] = Field(
        default_factory=list, description="참고한 문서 URL 목록"
    )


class ChatAnswer(BaseModel):
    answer: str = Field(description="질문에 대한 최종 답변")
    reference_urls: list[str] = Field(
        default_factory=list,
        description="답변 근거로 사용한 참고 문서 URL 목록. 없으면 빈 배열",
    )
