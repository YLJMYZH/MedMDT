# src/medmdt/extractor/parsers/image_parser.py
"""Multimodal LLM-based medical image analysis parser."""

import base64
from io import BytesIO
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from medmdt.llm.errors import (
    InvalidImageError,
    InvalidVisionResponse,
    VisionModelNotSupported,
    VisionRequestError,
)

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
DIRECT_IMAGE_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
PNG_COMPATIBLE_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}

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


def _prepare_image(image_data: bytes, max_image_bytes: int) -> tuple[bytes, str]:
    if not image_data:
        raise InvalidImageError("图片内容为空")
    invalid_image = False
    try:
        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            if image_format in DIRECT_IMAGE_MIME:
                image.verify()
                with Image.open(BytesIO(image_data)) as decoded_image:
                    decoded_image.load()
                normalized = image_data
                mime_type = DIRECT_IMAGE_MIME[image_format]
            else:
                image.load()
                if image.mode not in PNG_COMPATIBLE_MODES:
                    target_mode = "RGBA" if "A" in image.getbands() else "RGB"
                    image = image.convert(target_mode)
                output = BytesIO()
                image.save(output, format="PNG")
                normalized = output.getvalue()
                mime_type = "image/png"
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        invalid_image = True

    if invalid_image:
        raise InvalidImageError("图片损坏或格式无法识别")

    if len(normalized) > max_image_bytes:
        raise InvalidImageError(
            f"图片超过 {max_image_bytes} 字节传输大小限制；系统不会静默缩放医学影像"
        )
    return normalized, mime_type


def _is_image_input_rejection(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(
        response, "status_code", None
    )
    error_text = str(exc).lower()
    image_markers = ("image", "vision", "multimodal", "image_url", "图片", "视觉")
    return status_code in {400, 404, 422} and any(
        marker in error_text for marker in image_markers
    )


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    raise InvalidVisionResponse("视觉模型响应中没有可用文本")


def _invoke_model(llm: BaseChatModel, message: HumanMessage):
    try:
        return llm.invoke([message])
    except Exception as exc:
        image_input_rejected = _is_image_input_rejection(exc)

    # Raise outside the except scope so the SDK exception and any attached
    # Base64 medical image are not retained as context or cause.
    if image_input_rejected:
        raise VisionModelNotSupported("所选模型不接受图片输入")
    raise VisionRequestError("视觉模型请求失败，请检查认证、限流和服务状态")


class ImageParser:
    """Sends medical images to a multimodal LLM for structured analysis."""

    def __init__(
        self,
        llm: BaseChatModel,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ):
        self._llm = llm
        self._max_image_bytes = max_image_bytes

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
        normalized, mime_type = _prepare_image(image_data, self._max_image_bytes)
        for attempt in range(2):
            prompt = self._build_prompt(context)
            if attempt == 1:
                prompt += "\n\n上一次响应不是有效 JSON。请严格只返回符合上述字段定义的 JSON。"
            message = self._build_message(normalized, mime_type, prompt)
            response = _invoke_model(self._llm, message)
            raw = _extract_response_text(response.content)
            try:
                return self._parse_response(raw)
            except ValidationError:
                pass
        raise InvalidVisionResponse("视觉模型未返回有效的结构化 JSON")

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

    def _build_message(
        self, image_data: bytes, mime_type: str, prompt: str
    ) -> HumanMessage:
        encoded = base64.b64encode(image_data).decode("ascii")
        return HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ])

    @staticmethod
    def _parse_response(raw: str) -> ImageAnalysisResult:
        """Parse the LLM response, stripping markdown fences if present."""
        text = raw.strip()
        # Strip ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return ImageAnalysisResult.model_validate_json(text)
