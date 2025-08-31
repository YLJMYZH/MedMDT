# tests/test_ingest_script.py
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import pytest


def test_ingest_script_imports():
    from scripts.ingest import build_agent, run_ingest


@patch("scripts.ingest.ExtractionAgent")
@patch("scripts.ingest.KeywordStore")
@patch("scripts.ingest.VectorStore")
@patch("scripts.ingest.GraphStore")
@patch("scripts.ingest.create_vision_model")
@patch("scripts.ingest.create_chat_model")
@patch("scripts.ingest.load_llm_settings")
@patch("scripts.ingest.get_settings")
def test_build_agent_passes_independent_vision_model(
    get_settings,
    load_runtime,
    chat_factory,
    vision_factory,
    graph_cls,
    vector_cls,
    keyword_cls,
    agent_cls,
):
    from scripts.ingest import build_agent

    settings = MagicMock(
        default_llm_provider="openai",
        default_llm_model="text",
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="password",
        milvus_host="localhost",
        milvus_port=19530,
        embedding_dim=3,
        elasticsearch_url="http://localhost:9200",
    )
    get_settings.return_value = settings
    load_runtime.return_value = SimpleNamespace(
        vision=SimpleNamespace(
            provider="qwen", model="qwen-vl-max", api_key="k", base_url=None
        )
    )
    text_llm = MagicMock()
    vision_llm = MagicMock()
    chat_factory.return_value = text_llm
    vision_factory.return_value = vision_llm

    build_agent()

    vision_factory.assert_called_once_with("qwen", "qwen-vl-max", api_key="k")
    assert agent_cls.call_args.kwargs["llm"] is text_llm
    assert agent_cls.call_args.kwargs["vision_llm"] is vision_llm
    assert agent_cls.call_args.kwargs["vision_error"] is None


@patch("scripts.ingest.build_agent")
def test_run_ingest_single_file(mock_build, tmp_path):
    from scripts.ingest import run_ingest

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 5
    mock_report.relations_count = 3
    mock_report.chunks_count = 2
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    results = run_ingest(str(pdf))

    assert len(results) == 1
    mock_agent.process_file.assert_called_once_with(str(pdf))


@patch("scripts.ingest.build_agent")
def test_run_ingest_directory(mock_build, tmp_path):
    from scripts.ingest import run_ingest

    (tmp_path / "a.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "b.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "scan.webp").write_bytes(b"webp")
    (tmp_path / "scan.gif").write_bytes(b"gif")
    (tmp_path / "c.txt").write_text("not a pdf")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 1
    mock_report.relations_count = 0
    mock_report.chunks_count = 1
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    results = run_ingest(str(tmp_path))

    assert len(results) == 4
    assert mock_agent.process_file.call_count == 4


def test_cli_extensions_share_backend_image_extensions():
    from medmdt.extractor.file_types import IMAGE_EXTENSIONS
    from scripts.ingest import SUPPORTED_EXTENSIONS

    assert IMAGE_EXTENSIONS <= SUPPORTED_EXTENSIONS
