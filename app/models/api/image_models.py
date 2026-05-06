from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    index: int = Field(description="이미지 순서 (1부터 시작)")
    description: str = Field(description="이미지 설명")


class ImageDescriptions(BaseModel):
    images: list[ImageDescription] = Field(description="이미지 설명 리스트")
