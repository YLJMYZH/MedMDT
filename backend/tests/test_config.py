# tests/test_config.py
import os
import pytest
from medmdt.config.settings import Settings, get_settings


def test_settings_loads_defaults():
    s = Settings(paddleocr_token="test-token")
    assert s.neo4j_uri == "bolt://localhost:7687"
    assert s.milvus_host == "localhost"
    assert s.milvus_port == 19530
    assert s.elasticsearch_url == "http://localhost:9200"
    assert s.default_llm_provider == "openai"
    assert s.mdt_max_rounds == 3
    assert s.mdt_consensus_threshold == 0.8
    assert s.embedding_model == "bge-large-zh-v1.5"
    assert s.embedding_dim == 1024


def test_settings_requires_paddleocr_token():
    with pytest.raises(Exception):
        Settings()


def test_get_settings_returns_singleton():
    os.environ["MEDMDT_PADDLEOCR_TOKEN"] = "test"
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    del os.environ["MEDMDT_PADDLEOCR_TOKEN"]
