# tests/test_extraction_agent_dicom.py
import pytest
from unittest.mock import MagicMock
from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.parsers.dicom_parser import DicomParseResult, DicomMetadata
from medmdt.extractor.parsers.image_parser import ImageAnalysisResult
from medmdt.extractor.ingestor import IngestReport


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.paddleocr_api_url = "http://test"
    settings.paddleocr_token = "token"
    settings.paddleocr_use_doc_orientation_classify = False
    settings.paddleocr_use_doc_unwarping = False
    settings.paddleocr_use_chart_recognition = False
    return settings


@pytest.fixture
def agent(mock_settings):
    llm = MagicMock()
    return ExtractionAgent(
        settings=mock_settings,
        graph_store=MagicMock(),
        vector_store=MagicMock(),
        keyword_store=MagicMock(),
        embed_fn=MagicMock(return_value=[[0.1] * 1024]),
        llm=llm,
        vision_llm=llm,
    )


def test_process_dicom(agent):
    mock_parser = MagicMock()
    mock_parser.parse.return_value = DicomParseResult(
        metadata=DicomMetadata(
            patient_id="P001", patient_name="张三", modality="CT",
            study_description="胸部CT",
        ),
        image_analysis=ImageAnalysisResult(
            description="肺部见结节影",
            findings=["右肺结节", "直径约1cm"],
            modality="CT",
        ),
        raw_text="患者ID: P001\n患者姓名: 张三\n检查类型: CT\n影像分析: 肺部见结节影",
    )
    agent._dicom_parser = mock_parser

    # Mock the LLM response for entity extraction
    import json
    entity_response = MagicMock()
    entity_response.content = json.dumps({
        "entities": [{"name": "肺结节", "type": "disease", "aliases": []}],
        "relations": [],
    })
    chunk_response = MagicMock()
    chunk_response.content = json.dumps({
        "chunks": [{"text": "CT显示右肺结节", "summary": "右肺结节影", "keywords": ["结节", "CT"]}],
    })
    agent._llm.invoke.side_effect = [entity_response, chunk_response]

    reports = agent.process_file("/path/scan.dcm")
    assert len(reports) == 1
    assert isinstance(reports[0], IngestReport)
    mock_parser.parse.assert_called_once_with("/path/scan.dcm")


def test_process_dicom_no_image(agent):
    mock_parser = MagicMock()
    mock_parser.parse.return_value = DicomParseResult(
        metadata=DicomMetadata(patient_id="P002", patient_name="李四", modality="SR"),
        image_analysis=None,
        raw_text="患者ID: P002\n检查类型: SR",
    )
    agent._dicom_parser = mock_parser

    import json
    entity_response = MagicMock()
    entity_response.content = json.dumps({"entities": [], "relations": []})
    chunk_response = MagicMock()
    chunk_response.content = json.dumps({"chunks": [{"text": "SR报告", "summary": "结构化报告", "keywords": ["SR"]}]})
    agent._llm.invoke.side_effect = [entity_response, chunk_response]

    reports = agent.process_file("/path/report.dcm")
    assert len(reports) == 1
