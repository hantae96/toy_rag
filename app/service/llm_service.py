import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import base64
import logging
import requests
import numpy as np
import cv2
import re

from typing import AsyncGenerator, cast
from langchain.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith import traceable

from app.models.api.chat_models import ApiResponse, ChatAnswer
from app.models.api.image_models import ImageDescriptions

logger = logging.getLogger(__name__)

class LlmService:
    def __init__(
        self,
        chat_llm: Runnable,
        vision_llm: Runnable,
    ) -> None:
        self.llm = chat_llm
        self.vision_llm = vision_llm

        self.answer_parser = PydanticOutputParser(pydantic_object=ChatAnswer)

        self.structured_prompt = ChatPromptTemplate.from_template(
            """
                너는 사내 메뉴얼 문서만 근거로 답변하는 assistant다.
                아래 규칙을 바탕으로 질문에 답변해라.
                1. 참고 문서에 없는 내용은 추측하지 말고 종료해라.
                2. reference_urls에는 실제로 근거로 사용한 doc_url만 담아라.
                3. 답변 근거 URL이 없으면 reference_urls는 빈 배열로 반환해라.
                4. 만약 [이미지] 안에 내용을 참고하여 답변한다면, answer에 "해당 이미지를 직접 확인해주세요." 라는 문구를 추가해라.
                5. 반드시 아래 형식 지시를 따라 JSON으로만 응답해라.

                [형식 지시]
                {format_instructions}

                [참고 문서]
                {context}

                [질문]
                {question}
            """
        )
        self.structured_chain = (
            self.structured_prompt.partial(
                format_instructions=self.answer_parser.get_format_instructions()
            )
            | self.llm
            | self.answer_parser
        )

        self.stream_prompt = ChatPromptTemplate.from_template(
            """
                너는 사내 메뉴얼 문서만 근거로 답변하는 assistant다.
                아래 규칙을 바탕으로 질문에 답변해라.
                1. 참고 문서에 없는 내용은 추측하지 말고 종료해라.
                2. 출처가 존재하는 경우, 답변 마지막 줄에 반드시 문서 제목(title) + 링크(doc_url) 로 표기해라.
                3. 만약 [이미지] 안에 내용을 참고하여 답변한다면, "해당 이미지를 직접 확인해주세요." 라는 문구를 추가해라

                [참고 문서]
                {context}

                [질문]
                {question}
            """
        )
        self.stream_chain = self.stream_prompt | self.llm

    @traceable(name="llm_service.ask", run_type="chain")
    def ask(self, question: str, context: str) -> ApiResponse:
        try:
            parsed = cast(
                ChatAnswer,
                self.structured_chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    }
                ),
            )

            return ApiResponse(
                answer=parsed.answer,
                reference_urls=list(dict.fromkeys(parsed.reference_urls)),
            )
        
        except Exception as e:
            logger.warning("구조화 출력 파싱 실패, 일반 출력으로 fallback: %s", e)
            raw = self.stream_chain.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )
            raw_text = str(raw.content)
            urls = re.findall(r"https?://[^\s)\]]+", raw_text)
            return ApiResponse(
                answer=raw_text,
                reference_urls=list(dict.fromkeys(urls)),
            )

    async def ask_stream(
        self, question: str, context: str
    ) -> AsyncGenerator[str, None]:
        async for chunck in self.stream_chain.astream(
            {
                "context": context,
                "question": question,
            }
        ):
            if chunck.content:
                yield str(chunck.content)

    @traceable(name="llm_service.describe_images", run_type="chain")
    def describe_images(self, image_urls: list[str], batch_size: int = 5) -> list[str]:
        results: list[str] = []
        # for i in range(0, len(image_urls), batch_size):
        #     batch = image_urls[i : i + batch_size]
        #     descriptions = self._describe_images_batch(batch)
        #     results.extend(descriptions)
        return results

    @traceable(name="llm_service.describe_images_batch", run_type="chain")
    def _describe_images_batch(self, image_urls: list[str]) -> list[str]:
        try:
            content = []

            for image_url in image_urls:
                image_data = self._preprocess_image_from_url(image_url)

                if image_url.lower().endswith(".png"):
                    media_type = "image/png"
                elif image_url.lower().endswith(".gif"):
                    media_type = "image/gif"
                else:
                    media_type = "image/jpeg"

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                            "detail": "high",
                        },
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": """
                                회사의 매뉴얼에 부가설명을 위한 이미지 정보야.
                                이미지가 여러 장이면 각 이미지를 순서대로 설명해줘.

                                설명 방법
                                1. 이미지에 문자가 있으면 추출한 다음 해당 내용을 기반으로 이미지가 어떤 내용인지 포괄적으로 설명해줘.
                                    너무 자세한 숫자는 내용에 기입할 필요없고, 해당 이미지가 어떤 주제를 가지고 있는지를 분석해줘
                                2. 이미지가 표 형식인 경우, 표의 구조를 참조해서 이미지를 해석해줘
                                3. 이미지 내에 빨간 네모 박스 테두리는 강조를 위한 표시야.
                                4. 각 이미지 설명은 반드시 "[이미지 1]", "[이미지 2]" 형식으로 구분해줘.
                            """,
                }
            )

            result = cast(
                ImageDescriptions, self.vision_llm.invoke([HumanMessage(content=content)])
            )

            sorted_images = sorted(result.images, key=lambda x: x.index)
            descriptions = [img.description for img in sorted_images]

            if len(descriptions) != len(image_urls):
                return [""] * len(image_urls)

            return descriptions

        except Exception as e:
            logger.error(f"이미지 벌크 요약 실패 → {e}")
            return [""] * len(image_urls)

    
    def _preprocess_image_from_url(self, image_url: str) -> str:
        # 1. URL에서 이미지 바이트 다운로드
        response = requests.get(image_url, timeout=5, verify=False)
        if response.status_code != 200:
            raise ValueError(f"이미지 URL 호출에 실패했습니다. URL : {image_url}")

        image_bytes = np.frombuffer(response.content, dtype=np.uint8)

        # 2. OpenCV 이미지로 변환 (알파 채널 포함)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"이미지를 호출할 수 없습니다 : {image_url}")

        # 3. 투명 배경 → 흰 배경으로 합성
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
            img = (white_bg * (1 - alpha[..., None]) + img[:, :, :3] * alpha[..., None]).astype(np.uint8)

        # 4. base64 인코딩
        _, buffer = cv2.imencode('.png', img)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image
