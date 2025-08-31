# tests/test_extraction_agent.py
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.schemas import ParsedPage
from medmdt.extractor.ingestor import IngestReport
from medmdt.config.settings import Settings
from medmdt.llm.errors import VisionProviderNotSupported


@pytest.fixture
def settings():
    return Settings(paddleocr_token="test-token")


@pytest.fixture
def mock_deps():
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 1024])
    llm = MagicMock()
    return graph, vector, keyword, embed_fn, llm


def _mock_llm_responses(llm):
    entity_response = MagicMock()
    entity_response.content = '''{
        "entities": [{"name": "糖尿病", "type": "disease", "aliases": ["DM"]}],
        "relations": [{"head": "二甲双胍", "relation": "treats", "tail": "糖尿病", "evidence": "一线用药", "confidence": 0.95}]
    }'''

    chunk_response = MagicMock()
    chunk_response.content = '''{
        "chunks": [{"text": "糖尿病治疗", "summary": "治疗方案概述", "keywords": ["糖尿病", "治疗"], "metadata": {"section": "治疗"}}]
    }'''

    llm.invoke.side_effect = [entity_response, chunk_response]


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(output, format="PNG")
    return output.getvalue()


def test_extraction_agent_uses_separate_vision_model(settings, mock_deps):
    graph, vector, keyword, embed_fn, text_llm = mock_deps
    vision_llm = MagicMock()
    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=text_llm,
        vision_llm=vision_llm,
    )

    assert agent._llm is text_llm
    assert agent._image_parser is not None
    assert agent._image_parser._llm is vision_llm


def _agent_without_vision(settings, mock_deps, vision_error=None):
    graph, vector, keyword, embed_fn, text_llm = mock_deps
    return ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=text_llm,
        vision_llm=None,
        vision_error=vision_error,
    )


def test_process_image_raises_stable_configured_vision_error(
    settings, mock_deps, tmp_path
):
    agent = _agent_without_vision(
        settings, mock_deps, vision_error="DeepSeek 暂不支持图像理解"
    )
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(_png_bytes())

    with pytest.raises(
        VisionProviderNotSupported, match="DeepSeek 暂不支持图像理解"
    ):
        agent.process_file(str(image_path))


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_pdf_embedded_image_requires_vision(
    mock_ocr_cls, settings, mock_deps
):
    mock_ocr_cls.return_value.parse.return_value = [
        ParsedPage(page_num=0, markdown="text", images=[_png_bytes()])
    ]
    agent = _agent_without_vision(settings, mock_deps)

    with pytest.raises(
        VisionProviderNotSupported, match="图片分析未配置可用的视觉模型"
    ):
        agent.process_file("report.pdf")


def test_process_dicom_requires_vision(settings, mock_deps):
    agent = _agent_without_vision(settings, mock_deps)

    with pytest.raises(
        VisionProviderNotSupported, match="图片分析未配置可用的视觉模型"
    ):
        agent.process_file("scan.dcm")


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_text_only_pdf_succeeds_without_vision(
    mock_ocr_cls, settings, mock_deps
):
    graph, vector, keyword, embed_fn, text_llm = mock_deps
    _mock_llm_responses(text_llm)
    graph.upsert_entities.return_value = 1
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1
    mock_ocr_cls.return_value.parse.return_value = [
        ParsedPage(page_num=0, markdown="# 糖尿病指南", images=[])
    ]
    agent = _agent_without_vision(settings, mock_deps)

    reports = agent.process_file("text-only.pdf")

    assert len(reports) == 1
    assert isinstance(reports[0], IngestReport)
    assert text_llm.invoke.call_count == 2


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_pdf_file(mock_ocr_cls, settings, mock_deps):
    graph, vector, keyword, embed_fn, llm = mock_deps
    _mock_llm_responses(llm)

    graph.upsert_entities.return_value = 1
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    mock_ocr = MagicMock()
    mock_ocr.parse.return_value = [
        ParsedPage(page_num=0, markdown="# 糖尿病指南\n\n二甲双胍是一线用药", images=[]),
    ]
    mock_ocr_cls.return_value = mock_ocr

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    reports = agent.process_file("test.pdf")

    assert len(reports) == 1
    assert isinstance(reports[0], IngestReport)
    mock_ocr.parse.assert_called_once_with("test.pdf")


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_pdf_with_images_triggers_image_analysis(mock_ocr_cls, settings, mock_deps):
    graph, vector, keyword, embed_fn, llm = mock_deps

    entity_response = MagicMock()
    entity_response.content = '{"entities": [], "relations": []}'
    chunk_response = MagicMock()
    chunk_response.content = '{"chunks": [{"text": "t", "summary": "s", "keywords": [], "metadata": {}}]}'
    img_response = MagicMock()
    img_response.content = '{"description": "正常胸片", "findings": [], "modality": "X-ray"}'
    llm.invoke.side_effect = [img_response, entity_response, chunk_response]

    graph.upsert_entities.return_value = 0
    graph.upsert_relations.return_value = 0
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    mock_ocr = MagicMock()
    mock_ocr.parse.return_value = [
        ParsedPage(page_num=0, markdown="# 胸部X光报告", images=[_png_bytes()]),
    ]
    mock_ocr_cls.return_value = mock_ocr

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    reports = agent.process_file("xray_report.pdf")

    assert len(reports) == 1
    assert llm.invoke.call_count == 3  # image + entity + chunk


def test_detect_file_type():
    agent_cls = ExtractionAgent
    assert agent_cls._detect_file_type("report.pdf") == "pdf"
    assert agent_cls._detect_file_type("scan.dcm") == "dicom"
    assert agent_cls._detect_file_type("photo.jpg") == "image"
    assert agent_cls._detect_file_type("photo.png") == "image"
    assert agent_cls._detect_file_type("photo.jpeg") == "image"


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_classify_document_guideline(mock_ocr_cls, settings, mock_deps):
    """Verify _classify_document detects clinical guidelines."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    assert agent._classify_document("这是一份临床指南文档") == "clinical_guideline"
    assert agent._classify_document("guideline for treatment") == "clinical_guideline"
    assert agent._classify_document("专家共识") == "clinical_guideline"


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_classify_document_case_report(mock_ocr_cls, settings, mock_deps):
    """Verify _classify_document detects case reports."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    assert agent._classify_document("患者入院记录") == "case_report"
    assert agent._classify_document("case study of patient") == "case_report"


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_classify_document_textbook(mock_ocr_cls, settings, mock_deps):
    """Verify _classify_document detects textbook content."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    assert agent._classify_document("内科学教材第三章") == "textbook"


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_classify_document_other(mock_ocr_cls, settings, mock_deps):
    """Verify _classify_document falls back to 'other'."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    assert agent._classify_document("随机文本") == "other"


def test_detect_file_type_unknown_raises():
    """Unsupported extensions should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown file extension"):
        ExtractionAgent._detect_file_type("file.xyz")


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_dicom_delegates_to_dicom_parser(mock_ocr_cls, settings, mock_deps):
    """DICOM processing delegates to DicomParser and returns IngestReport."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )

    # Mock the dicom parser directly on the agent instance
    mock_dicom_parser = MagicMock()
    from medmdt.extractor.parsers.dicom_parser import DicomParseResult, DicomMetadata
    mock_dicom_parser.parse.return_value = DicomParseResult(
        metadata=DicomMetadata(patient_id="P001", patient_name="Test", modality="CT"),
        image_analysis=None,
        raw_text="患者ID: P001\n检查类型: CT",
    )
    agent._dicom_parser = mock_dicom_parser

    import json
    entity_resp = MagicMock()
    entity_resp.content = json.dumps({"entities": [], "relations": []})
    chunk_resp = MagicMock()
    chunk_resp.content = json.dumps({"chunks": [{"text": "CT scan", "summary": "CT", "keywords": ["CT"]}]})
    llm.invoke.side_effect = [entity_resp, chunk_resp]

    reports = agent.process_file("scan.dcm")
    assert len(reports) == 1
    mock_dicom_parser.parse.assert_called_once_with("scan.dcm")


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_extract_from_text_parses_entities_and_chunks(mock_ocr_cls, settings, mock_deps):
    """Verify _extract_from_text correctly parses LLM JSON into structured objects."""
    graph, vector, keyword, embed_fn, llm = mock_deps
    mock_ocr_cls.return_value = MagicMock()

    entity_response = MagicMock()
    entity_response.content = '''{
        "entities": [
            {"name": "高血压", "type": "disease", "aliases": ["HTN"]},
            {"name": "头痛", "type": "symptom", "aliases": []}
        ],
        "relations": [
            {"head": "高血压", "relation": "causes", "tail": "头痛", "evidence": "高血压可导致头痛", "confidence": 0.85}
        ]
    }'''
    chunk_response = MagicMock()
    chunk_response.content = '''{
        "chunks": [
            {"text": "高血压可导致头痛", "summary": "高血压的症状", "keywords": ["高血压", "头痛"], "metadata": {"section": "临床表现"}}
        ]
    }'''
    llm.invoke.side_effect = [entity_response, chunk_response]

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    from medmdt.extractor.schemas import SourceInfo
    result = agent._extract_from_text(
        text="高血压可导致头痛",
        source=SourceInfo(file="test.pdf", type="other"),
    )

    assert len(result.entities) == 2
    assert result.entities[0].name == "高血压"
    assert result.entities[0].type == "disease"
    assert len(result.relations) == 1
    assert result.relations[0].relation == "causes"
    assert len(result.chunks) == 1
    assert result.chunks[0].summary == "高血压的症状"


@patch("medmdt.extractor.agent.PaddleOCRClient")
def test_process_pdf_multi_page(mock_ocr_cls, settings, mock_deps):
    """Multiple pages produce multiple ingest reports."""
    graph, vector, keyword, embed_fn, llm = mock_deps

    # Two pages -> 4 LLM calls (entity+chunk per page)
    responses = []
    for _ in range(2):
        er = MagicMock()
        er.content = '{"entities": [], "relations": []}'
        cr = MagicMock()
        cr.content = '{"chunks": [{"text": "t", "summary": "s", "keywords": [], "metadata": {}}]}'
        responses.extend([er, cr])
    llm.invoke.side_effect = responses

    graph.upsert_entities.return_value = 0
    graph.upsert_relations.return_value = 0
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    mock_ocr = MagicMock()
    mock_ocr.parse.return_value = [
        ParsedPage(page_num=0, markdown="Page 1 content", images=[]),
        ParsedPage(page_num=1, markdown="Page 2 content", images=[]),
    ]
    mock_ocr_cls.return_value = mock_ocr

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=llm,
    )
    reports = agent.process_file("multi.pdf")

    assert len(reports) == 2
    assert llm.invoke.call_count == 4
