# tests/test_image_parser.py
from unittest.mock import MagicMock
import pytest
from medmdt.extractor.parsers.image_parser import ImageParser, ImageAnalysisResult


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
    result = parser.analyze(b"fake-image-bytes")

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
    result = parser.analyze(b"fake-image", context="患者咳嗽2周")

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
    results = parser.analyze_batch([b"img1", b"img2"])

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
    image_bytes = b"PNG-fake-data"
    parser.analyze(image_bytes)

    call_args = mock_model.invoke.call_args[0][0]
    message = call_args[0]
    # The message content should contain image_url with base64 data
    image_part = [p for p in message.content if p.get("type") == "image_url"][0]
    expected_b64 = base64.b64encode(image_bytes).decode()
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
    result = parser.analyze(b"img")

    assert result.description == "test"
    assert result.modality == "CT"
