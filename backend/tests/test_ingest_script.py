# tests/test_ingest_script.py
import asyncio
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
    monkeypatch,
):
    from scripts.ingest import build_agent

    monkeypatch.setenv("OPENAI_API_KEY", "cli-openai-key")

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


@patch("scripts.ingest.ExtractionAgent")
@patch("scripts.ingest.KeywordStore")
@patch("scripts.ingest.VectorStore")
@patch("scripts.ingest.GraphStore")
@patch("scripts.ingest.create_vision_model")
@patch("scripts.ingest.create_chat_model")
@patch("scripts.ingest.load_llm_settings")
@patch("scripts.ingest.get_settings")
def test_build_agent_revalidates_stale_runtime_vision_before_construction(
    get_settings,
    load_runtime,
    chat_factory,
    vision_factory,
    graph_cls,
    vector_cls,
    keyword_cls,
    agent_cls,
    monkeypatch,
):
    from scripts.ingest import build_agent

    monkeypatch.delenv("MEDMDT_ALLOWED_BASE_URL_HOSTS", raising=False)
    get_settings.return_value = MagicMock(
        default_llm_provider="openai",
        default_llm_model="text",
    )
    load_runtime.return_value = SimpleNamespace(
        vision=SimpleNamespace(
            provider="custom",
            model="legacy-vision",
            api_key="legacy-key",
            base_url="https://removed-cli.example/v1",
        )
    )

    with pytest.raises(ValueError, match="not allowed"):
        build_agent()

    vision_factory.assert_not_called()
    graph_cls.assert_not_called()
    vector_cls.assert_not_called()
    keyword_cls.assert_not_called()
    agent_cls.assert_not_called()


@patch("scripts.ingest.ExtractionAgent")
@patch("scripts.ingest.KeywordStore")
@patch("scripts.ingest.VectorStore")
@patch("scripts.ingest.GraphStore")
@patch("scripts.ingest.create_vision_model")
@patch("scripts.ingest.create_chat_model")
@patch("scripts.ingest.load_llm_settings")
@patch("scripts.ingest.get_settings")
def test_cli_embeddings_borrow_shared_redirect_safe_clients(
    get_settings,
    load_runtime,
    chat_factory,
    vision_factory,
    graph_cls,
    vector_cls,
    keyword_cls,
    agent_cls,
    monkeypatch,
):
    from medmdt.llm.http_clients import (
        close_shared_http_clients,
        get_shared_async_http_client,
        get_shared_http_client,
    )
    from scripts.ingest import build_agent

    monkeypatch.setenv("OPENAI_API_KEY", "cli-openai-key")

    settings = MagicMock(
        default_llm_provider="openai",
        default_llm_model="text",
        embedding_model="embed",
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
    asyncio.run(close_shared_http_clients())
    try:
        with patch("langchain_openai.OpenAIEmbeddings") as embedding_cls:
            build_agent()
            embed_fn = agent_cls.call_args.kwargs["embed_fn"]
            embed_fn(["document"])

        kwargs = embedding_cls.call_args.kwargs
        assert kwargs["http_client"] is get_shared_http_client()
        assert kwargs["http_async_client"] is get_shared_async_http_client()
        assert kwargs["http_client"].follow_redirects is False
        assert kwargs["http_async_client"].follow_redirects is False
        assert kwargs["openai_api_base"] == "https://api.openai.com/v1"
        assert kwargs["api_key"] == "cli-openai-key"
    finally:
        asyncio.run(close_shared_http_clients())


@patch("scripts.ingest.build_agent")
def test_run_ingest_single_file(mock_build, tmp_path):
    from scripts.ingest import run_ingest
    from medmdt.llm.http_clients import (
        close_shared_http_clients,
        get_shared_async_http_client,
        get_shared_http_client,
    )

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 5
    mock_report.relations_count = 3
    mock_report.chunks_count = 2
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    asyncio.run(close_shared_http_clients())
    sync_client = get_shared_http_client()
    async_client = get_shared_async_http_client()
    try:
        results = run_ingest(str(pdf))

        assert len(results) == 1
        mock_agent.process_file.assert_called_once_with(str(pdf))
        assert sync_client.is_closed
        assert async_client.is_closed
    finally:
        asyncio.run(close_shared_http_clients())


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
