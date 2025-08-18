# src/medmdt/config/settings.py
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "MEDMDT_", "env_file": ".env", "extra": "ignore"}

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    milvus_host: str = "localhost"
    milvus_port: int = 19530

    elasticsearch_url: str = "http://localhost:9200"

    paddleocr_api_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    paddleocr_use_doc_orientation_classify: bool = False
    paddleocr_use_doc_unwarping: bool = False
    paddleocr_use_chart_recognition: bool = False

    mdt_max_rounds: int = 3
    mdt_consensus_threshold: float = 0.8

    embedding_model: str = "bge-large-zh-v1.5"
    embedding_dim: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
