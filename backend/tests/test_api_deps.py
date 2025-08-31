import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.datastructures import UploadFile

from medmdt.llm.errors import (
    VisionProviderNotSupported,
    VisionRequestError,
)


@patch("medmdt.config.settings.get_settings")
@patch("medmdt.config.runtime.load_llm_settings")
@patch("medmdt.llm.provider.create_vision_model")
@patch("medmdt.llm.provider.create_chat_model")
@patch("medmdt.knowledge.retriever.FusionRetriever")
@patch("medmdt.knowledge.keyword_store.KeywordStore")
@patch("medmdt.knowledge.vector_store.VectorStore")
@patch("medmdt.knowledge.graph_store.GraphStore")
def test_build_infrastructure_keeps_running_when_legacy_vision_is_unsupported(
    graph_cls,
    vector_cls,
    keyword_cls,
    retriever_cls,
    chat_factory,
    vision_factory,
    load_runtime,
    get_settings,
):
    from medmdt.api.deps import build_infrastructure

    get_settings.return_value = MagicMock(
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="password",
        milvus_host="localhost",
        milvus_port=19530,
        elasticsearch_url="http://localhost:9200",
    )
    load_runtime.return_value = SimpleNamespace(
        knowledge=SimpleNamespace(
            provider="openai", model="text", api_key="k", base_url=None
        ),
        vision=SimpleNamespace(
            provider="deepseek",
            model="deepseek-chat",
            api_key="k",
            base_url=None,
        ),
        embedding=SimpleNamespace(
            provider="openai",
            model="embed",
            api_key="k",
            base_url=None,
            dim=3,
        ),
    )
    chat_factory.return_value = MagicMock()
    vision_factory.side_effect = VisionProviderNotSupported(
        "DeepSeek 暂不支持图像理解"
    )

    infra = build_infrastructure()

    assert infra["vision_llm"] is None
    assert infra["vision_error"] == "DeepSeek 暂不支持图像理解"


def _infra(vision_llm=None):
    return {
        "settings": MagicMock(),
        "llm": MagicMock(),
        "vision_llm": vision_llm,
        "vision_error": "DeepSeek 暂不支持图像理解" if vision_llm is None else None,
        "vision_provider": "deepseek" if vision_llm is None else "qwen",
        "vision_model": "deepseek-chat" if vision_llm is None else "qwen-vl-max",
        "graph_store": MagicMock(),
        "vector_store": MagicMock(),
        "keyword_store": MagicMock(),
        "embed_fn": MagicMock(),
    }


@patch("medmdt.extractor.agent.ExtractionAgent")
@patch("medmdt.api.deps.build_infrastructure")
def test_run_ingest_passes_vision_dependencies(
    build_infrastructure, agent_cls, tmp_path
):
    from medmdt.api.routes.knowledge import _ingest_jobs, _run_ingest

    infra = _infra(vision_llm=MagicMock())
    build_infrastructure.return_value = infra
    agent_cls.return_value.process_file.return_value = []
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF")
    _ingest_jobs["job-1"] = {"status": "queued"}

    _run_ingest("job-1", str(file_path))

    assert agent_cls.call_args.kwargs["vision_llm"] is infra["vision_llm"]
    assert agent_cls.call_args.kwargs["vision_error"] is None


@patch("medmdt.extractor.agent.ExtractionAgent")
@patch("medmdt.api.deps.build_infrastructure")
def test_run_ingest_logs_only_safe_metadata_for_vision_errors(
    build_infrastructure, agent_cls, tmp_path, caplog
):
    from medmdt.api.routes.knowledge import _ingest_jobs, _run_ingest

    infra = _infra()
    infra["settings"].api_key = "secret-key"
    build_infrastructure.return_value = infra
    agent_cls.return_value.process_file.side_effect = VisionRequestError(
        "raw SDK error with clinical context"
    )
    file_path = tmp_path / "patient-secret.png"
    file_path.write_bytes(b"secret-image-bytes")
    _ingest_jobs["job-safe"] = {"status": "queued"}
    caplog.set_level("WARNING", logger="medmdt.api.routes.knowledge")

    _run_ingest("job-safe", str(file_path))

    assert _ingest_jobs["job-safe"]["message"] == "raw SDK error with clinical context"
    record = caplog.records[-1]
    assert record.getMessage() == (
        "Vision ingest failed job_id=job-safe provider=deepseek "
        "model=deepseek-chat error_type=VisionRequestError"
    )
    assert record.exc_info is None
    assert "secret-key" not in caplog.text
    assert "patient-secret" not in caplog.text
    assert "secret-image-bytes" not in caplog.text
    assert "clinical context" not in caplog.text


@pytest.mark.parametrize("suffix", [".png", ".dcm"])
def test_consultation_image_extraction_requires_vision(suffix):
    from medmdt.api.routes.consultation import _extract_text_from_file

    with pytest.raises(
        VisionProviderNotSupported, match="图片分析未配置可用的视觉模型"
    ):
        _extract_text_from_file("unused", suffix, MagicMock(), None)


@patch("medmdt.extractor.parsers.image_parser.ImageParser")
def test_consultation_image_extraction_preserves_stable_vision_error(
    image_parser_cls, tmp_path
):
    from medmdt.api.routes.consultation import _extract_text_from_file

    image_parser_cls.return_value.analyze.side_effect = VisionRequestError(
        "视觉服务暂不可用"
    )
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"image")

    with pytest.raises(VisionRequestError, match="视觉服务暂不可用"):
        _extract_text_from_file(str(image_path), ".png", MagicMock(), MagicMock())


@patch("medmdt.extractor.parsers.image_parser.ImageParser")
def test_consultation_image_extraction_hides_unexpected_errors(
    image_parser_cls, tmp_path
):
    from medmdt.api.routes.consultation import _extract_text_from_file

    image_parser_cls.return_value.analyze.side_effect = RuntimeError(
        "raw SDK error with clinical context"
    )
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"image")

    with pytest.raises(ValueError, match="^图片分析失败$"):
        _extract_text_from_file(str(image_path), ".png", MagicMock(), MagicMock())


@patch("medmdt.api.deps.build_infrastructure")
def test_consultation_upload_logs_only_safe_metadata(build_infrastructure, caplog):
    from medmdt.api.routes.consultation import upload_consultation_files

    build_infrastructure.return_value = _infra()
    upload = UploadFile(filename="patient-secret.png", file=BytesIO(b"secret-image"))
    caplog.set_level("WARNING", logger="medmdt.api.routes.consultation")

    response = asyncio.run(upload_consultation_files([upload]))

    assert response["records"][0]["error"] == "图片分析未配置可用的视觉模型"
    record = caplog.records[-1]
    message = record.getMessage()
    assert message.startswith("Vision upload failed request_id=")
    assert "provider=deepseek" in message
    assert "model=deepseek-chat" in message
    assert "error_type=VisionProviderNotSupported" in message
    assert record.exc_info is None
    assert "patient-secret" not in caplog.text
    assert "secret-image" not in caplog.text
    assert "DeepSeek 暂不支持" not in caplog.text
