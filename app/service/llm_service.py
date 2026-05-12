import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import base64
import logging
import time
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
                너는 사내 매뉴얼 문서만 근거로 답변하는 assistant다.
                아래 규칙을 엄격히 준수하여 답변해라.

                [답변 규칙]
                1. 반드시 [참고 문서] 내용만 근거로 답변하고, 문서에 없는 내용은 "제공된 매뉴얼에서 해당 내용을 찾을 수 없습니다."라고 답한 후 종료해라.
                2. 추측, 일반 상식, 외부 지식을 절대 사용하지 마라.
                3. 답변은 간결하고 구조적으로 작성하되, 절차/단계가 있으면 번호 매기기로 표기해라.
                4. 전문용어나 약어가 있으면 매뉴얼에 정의된 의미를 그대로 사용해라.

                [이미지 참조 규칙]
                5. 참고 문서의 [이미지] 내용을 근거로 답변한 경우, 답변 본문 마지막에 "※ 정확한 화면은 해당 이미지를 직접 확인해주세요." 문구를 추가해라.

                [출처 표기 규칙]
                6. 출처가 존재하는 경우, 답변 가장 마지막 줄에 아래 형식으로 표기해라.
                  형식: 출처: [문서제목](doc_url)
                7. 여러 문서를 참고한 경우 모두 나열해라.

                [참고 문서]
                {context}

                [질문]
                {question}

                [답변]
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
                너는 사내 매뉴얼 문서만 근거로 답변하는 assistant다.
                아래 규칙을 엄격히 준수하여 답변해라.

                [답변 규칙]
                1. 반드시 [참고 문서] 내용만 근거로 답변하고, 문서에 없는 내용은 "제공된 매뉴얼에서 해당 내용을 찾을 수 없습니다."라고 답한 후 종료해라.
                2. 추측, 일반 상식, 외부 지식을 절대 사용하지 마라.
                3. 답변은 간결하고 구조적으로 작성하되, 절차/단계가 있으면 번호 매기기로 표기해라.
                4. 전문용어나 약어가 있으면 매뉴얼에 정의된 의미를 그대로 사용해라.

                [이미지 참조 규칙]
                5. 참고 문서의 [이미지] 내용을 근거로 답변한 경우, 답변 본문 마지막에 "※ 정확한 화면은 해당 이미지를 직접 확인해주세요." 문구를 추가해라.

                [출처 표기 규칙]
                6. 출처가 존재하는 경우, 답변 가장 마지막 줄에 아래 형식으로 표기해라.
                  형식: 출처: [문서제목](doc_url)
                7. 여러 문서를 참고한 경우 모두 나열해라.

                [참고 문서]
                {context}

                [질문]
                {question}

                [답변]
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
    def describe_images(self, image_urls: list[str]) -> list[str]:
        return [self._describe_single_image(url) for url in image_urls]

    def _describe_single_image(self, image_url: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                image_data = self._preprocess_image_from_url(image_url)

                if image_url.lower().endswith(".png"):
                    media_type = "image/png"
                elif image_url.lower().endswith(".gif"):
                    media_type = "image/gif"
                else:
                    media_type = "image/jpeg"

                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": """
                            당신은 사내 업무 매뉴얼의 스크린샷을 분석하는 전문가입니다.
                            첨부된 이미지가 "어떤 화면이고 무엇을 하는 화면인지"만 간결하게 설명해주세요.

                            # 출력 항목

                            ## 1. 화면 식별
                            - 시스템/프로그램명: (예: SAP, 자체 ERP, MES 등 추정)
                            - 화면명: 상단 타이틀/탭에 표시된 명칭

                            ## 2. 화면 요약
                            - 이 화면이 어떤 업무를 위한 것인지 2~3문장으로 요약
                            - 화면에 표시된 핵심 데이터의 주제 (예: 자재 마스터, 구매 단가 정보 등)

                            # 작성 규칙
                            - 한국어로 작성
                            - 세부 버튼, 입력 필드, 레이아웃 설명은 생략
                            - 추측인 경우 "추정"으로 표기
                            - 식별 불가능한 항목은 "확인 불가"로 표기
                            - 약어(SAP, ROH 등)는 괄호로 부연
                        """,
                    },
                ]

                result = cast(
                    ImageDescriptions, self.vision_llm.invoke([HumanMessage(content=content)])
                )

                if result.images:
                    return result.images[0].description
                return ""

            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Rate limit 초과, %d초 후 재시도 (%d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    logger.error("이미지 설명 실패: %s → %s", image_url, e)
                    return ""
        return ""

    
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
            alpha = img[:, :, 3].astype(np.float64) / 255.0
            white_bg = np.full_like(img[:, :, :3], 255, dtype=np.float64)
            img = (white_bg * (1 - alpha[..., None]) + img[:, :, :3].astype(np.float64) * alpha[..., None]).astype(np.uint8)

        # 4. base64 인코딩
        _, buffer = cv2.imencode('.png', img)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image
