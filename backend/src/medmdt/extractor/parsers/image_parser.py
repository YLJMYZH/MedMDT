# src/medmdt/extractor/parsers/image_parser.py
"""Multimodal LLM-based medical image analysis parser."""

import base64
import json
import re

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

ANALYSIS_PROMPT = """你是一位资深医学影像分析专家。请分析这张医学图像，输出JSON格式：
{
  "description": "影像的整体描述",
  "findings": ["发现1", "发现2", ...],
  "modality": "影像类型，如X-ray/CT/MRI/病理切片/超声等，无法判断则为null"
}

只输出JSON，不要其他文字。"""


class ImageAnalysisResult(BaseModel):
    """Structured result from multimodal LLM image analysis."""

    description: str
    findings: list[str] = Field(default_factory=list)
    modality: str | None = None


class ImageParser:
    """Sends medical images to a multimodal LLM for structured analysis."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def analyze(
        self, image_data: bytes, context: str = ""
    ) -> ImageAnalysisResult:
        """Analyze a single medical image.

        Args:
            image_data: Raw image bytes (PNG, JPEG, etc.).
            context: Optional clinical context to guide analysis.

        Returns:
            Structured ImageAnalysisResult.
        """
        b64 = base64.b64encode(image_data).decode()
        content: list[dict] = [
            {"type": "text", "text": self._build_prompt(context)},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
        message = HumanMessage(content=content)
        response = self._llm.invoke([message])
        return self._parse_response(response.content)

    def analyze_batch(
        self, images: list[bytes], context: str = ""
    ) -> list[ImageAnalysisResult]:
        """Analyze multiple images sequentially.

        Args:
            images: List of raw image byte buffers.
            context: Optional clinical context shared across all images.

        Returns:
            List of ImageAnalysisResult, one per image.
        """
        return [self.analyze(img, context) for img in images]

    def _build_prompt(self, context: str) -> str:
        if context:
            return f"{ANALYSIS_PROMPT}\n\n临床背景：{context}"
        return ANALYSIS_PROMPT

    @staticmethod
    def _parse_response(raw: str) -> ImageAnalysisResult:
        """Parse the LLM response, stripping markdown fences if present."""
        text = raw.strip()
        # Strip ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return ImageAnalysisResult.model_validate_json(text)
