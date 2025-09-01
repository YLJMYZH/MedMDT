# tests/test_config.py
from medmdt.config.settings import Settings, get_settings


def test_settings_loads_defaults():
    settings = Settings()

    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.milvus_host == "localhost"
    assert settings.milvus_port == 19530
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.paddleocr_api_url == (
        "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    )
    assert settings.mdt_max_rounds == 3
    assert settings.mdt_consensus_threshold == 0.8


def test_settings_starts_without_paddleocr_token():
    settings = Settings()

    assert "paddleocr_token" not in type(settings).model_fields


def test_get_settings_returns_singleton():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
