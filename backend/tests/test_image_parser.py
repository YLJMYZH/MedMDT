# tests/test_image_parser.py
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from medmdt.extractor.parsers.image_parser import ImageParser, ImageAnalysisResult
from medmdt.llm.errors import (
    InvalidImageError,
    InvalidVisionResponse,
    VisionModelNotSupported,
    VisionRequestError,
)


def image_bytes(fmt: str, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(output, format=fmt)
    return output.getvalue()


def test_image_analysis_result():
    r = ImageAnalysisResult(
        description="右肺中叶见斑片状高密度影",
        findings=["右肺中叶炎症", "心影大小正常"],
        modality="X-ray",
    )
    assert len(r.findings) == 2


def test_image_analysis_result_defaults():
    r = ImageAnalysisResult(description="normal")
    assert r.findings == []
    assert r.modality is None


def test_image_parser_init():
    mock_model = MagicMock()
    parser = ImageParser(llm=mock_model)
    assert parser._llm is mock_model


def test_image_parser_analyze():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"description": "正常胸片", "findings": ["无明显异常"], "modality": "X-ray"}'
    )
    mock_model.invoke.return_value = mock_response

    parser = ImageParser(llm=mock_model)
    result = parser.analyze(image_bytes("PNG"))

    assert isinstance(result, ImageAnalysisResult)
    assert result.description == "正常胸片"
    mock_model.invoke.assert_called_once()


def test_image_parser_analyze_with_context():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"description": "肺部CT", "findings": ["左下肺结节"], "modality": "CT"}'
    )
    mock_model.invoke.return_value = mock_response

    parser = ImageParser(llm=mock_model)
    result = parser.analyze(image_bytes("PNG"), context="患者咳嗽2周")

    assert "结节" in result.findings[0]
    call_args = mock_model.invoke.call_args[0][0]
    message_contents = str(call_args)
    assert "咳嗽" in message_contents


def test_image_parser_analyze_batch():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"description": "desc", "findings": ["f1"], "modality": "X-ray"}'
    )
    mock_model.invoke.return_value = mock_response

    parser = ImageParser(llm=mock_model)
    results = parser.analyze_batch([image_bytes("PNG"), image_bytes("JPEG")])

    assert len(results) == 2
    assert mock_model.invoke.call_count == 2


def test_image_parser_sends_base64_image():
    """Verify the message sent to LLM contains base64-encoded image data."""
    import base64

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"description": "test", "findings": [], "modality": null}'
    )
    mock_model.invoke.return_value = mock_response

    parser = ImageParser(llm=mock_model)
    image_data = image_bytes("PNG")
    parser.analyze(image_data)

    call_args = mock_model.invoke.call_args[0][0]
    message = call_args[0]
    # The message content should contain image_url with base64 data
    image_part = [p for p in message.content if p.get("type") == "image_url"][0]
    expected_b64 = base64.b64encode(image_data).decode()
    assert expected_b64 in image_part["image_url"]["url"]


def test_image_parser_analyze_json_with_markdown_fence():
    """Parser should handle LLM responses wrapped in markdown code fences."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '```json\n{"description": "test", "findings": ["a"], "modality": "CT"}\n```'
    )
    mock_model.invoke.return_value = mock_response

    parser = ImageParser(llm=mock_model)
    result = parser.analyze(image_bytes("PNG"))

    assert result.description == "test"
    assert result.modality == "CT"


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
        ("GIF", "image/gif"),
    ],
)
def test_analyze_uses_detected_mime(fmt, mime):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"description":"ok","findings":[],"modality":null}'
    )
    ImageParser(llm).analyze(image_bytes(fmt))
    block = llm.invoke.call_args.args[0][0].content[1]
    assert block["image_url"]["url"].startswith(f"data:{mime};base64,")


def test_bmp_is_converted_to_png_before_sending():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"description":"ok","findings":[],"modality":null}'
    )
    ImageParser(llm).analyze(image_bytes("BMP"))
    block = llm.invoke.call_args.args[0][0].content[1]
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("payload", [b"", b"not-an-image"])
def test_invalid_images_fail_before_model_call(payload):
    llm = MagicMock()
    with pytest.raises(InvalidImageError):
        ImageParser(llm).analyze(payload)
    llm.invoke.assert_not_called()


def test_transmission_size_limit_does_not_resize():
    llm = MagicMock()
    with pytest.raises(InvalidImageError, match="大小限制"):
        ImageParser(llm, max_image_bytes=10).analyze(image_bytes("PNG"))
    llm.invoke.assert_not_called()


def test_content_block_response_is_normalized_to_text():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=[
        {"type": "thinking", "thinking": "private"},
        {
            "type": "text",
            "text": '{"description":"ok","findings":[],"modality":null}',
        },
    ])
    result = ImageParser(llm).analyze(image_bytes("PNG"))
    assert result.description == "ok"


def test_invalid_json_is_retried_once_with_same_image():
    llm = MagicMock()
    llm.invoke.side_effect = [
        AIMessage(content="not json"),
        AIMessage(content='{"description":"fixed","findings":[],"modality":null}'),
    ]
    result = ImageParser(llm).analyze(image_bytes("PNG"))
    assert result.description == "fixed"
    assert llm.invoke.call_count == 2
    assert "上一次响应不是有效 JSON" in llm.invoke.call_args.args[0][0].content[0]["text"]


def test_second_invalid_json_raises_stable_error():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="not json")
    with pytest.raises(InvalidVisionResponse):
        ImageParser(llm).analyze(image_bytes("PNG"))
    assert llm.invoke.call_count == 2


def test_image_input_rejection_is_classified_as_unsupported():
    llm = MagicMock()
    error = RuntimeError("image_url input is unsupported")
    error.status_code = 400
    llm.invoke.side_effect = error

    with pytest.raises(VisionModelNotSupported, match="不接受图片输入"):
        ImageParser(llm).analyze(image_bytes("PNG"))


def test_other_model_errors_raise_stable_request_error():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("secret provider details")

    with pytest.raises(VisionRequestError, match="视觉模型请求失败"):
        ImageParser(llm).analyze(image_bytes("PNG"))
