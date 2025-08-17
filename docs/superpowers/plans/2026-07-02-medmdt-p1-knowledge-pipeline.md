# MedMDT P1: Knowledge Extraction & Retrieval Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the knowledge extraction pipeline (PDF → structured JSON) and the triple-store knowledge base (Neo4j + Milvus + Elasticsearch) with unified fusion retrieval, so that medical documents can be ingested and searched.

**Architecture:** Offline extraction agent processes PDF files via PaddleOCR-VL-1.6 API, then uses an LLM to extract entities/relations/text chunks into a unified JSON schema. An ingestor writes entities+relations to Neo4j, text embeddings to Milvus, and full-text+keywords to Elasticsearch. A fusion retriever combines all three search paths via Reciprocal Rank Fusion (RRF).

**Tech Stack:** Python 3.12, uv, LangGraph, LangChain (langchain-openai, langchain-anthropic, langchain-ollama), Neo4j (neo4j driver), Milvus (pymilvus), Elasticsearch (elasticsearch-py), PaddleOCR-VL-1.6 cloud API, Pydantic v2, pydantic-settings, PyYAML, Pillow, pydicom, pytest

## Global Constraints

- Python >=3.12, managed by uv
- All config via pydantic-settings; secrets from environment variables, never hardcoded
- Source layout: `src/medmdt/` (src-layout)
- LLM providers: openai, anthropic, qwen, zhipu, moonshot, deepseek, ollama, custom — all via LangChain `BaseChatModel`
- Qwen/Zhipu/Moonshot/DeepSeek/Custom all use `langchain-openai`'s `ChatOpenAI` with provider-specific `base_url` (they all expose OpenAI-compatible APIs)
- Docker Compose for Neo4j 5, Milvus 2.4, Elasticsearch 8.15 (with IK plugin)
- Tests: pytest, use mocks for external services in unit tests

---

### Task 1: Project Foundation — Config, Dependencies, Docker

**Files:**
- Modify: `pyproject.toml`
- Create: `src/medmdt/__init__.py`
- Create: `src/medmdt/config/__init__.py`
- Create: `src/medmdt/config/settings.py`
- Create: `config/experts.yaml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `get_settings() -> Settings` — singleton accessor used by every other module. `Settings` fields: `neo4j_uri`, `neo4j_user`, `neo4j_password`, `milvus_host`, `milvus_port`, `elasticsearch_url`, `default_llm_provider`, `default_llm_model`, `paddleocr_api_url`, `paddleocr_token`, `paddleocr_use_doc_orientation_classify`, `paddleocr_use_doc_unwarping`, `paddleocr_use_chart_recognition`, `mdt_max_rounds`, `mdt_consensus_threshold`, `embedding_model`, `embedding_dim`

- [ ] **Step 1: Update pyproject.toml with all P1 dependencies**

```toml
[project]
name = "medmdt"
version = "0.1.0"
description = "Multi-expert medical consultation agent system"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.2.7",
    "langchain-core>=0.3",
    "langchain-openai>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-ollama>=0.3",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "neo4j>=5.0",
    "pymilvus>=2.4",
    "elasticsearch>=8.0",
    "requests>=2.31",
    "pillow>=10.0",
    "pydicom>=2.4",
    "pytest>=8.0",
]
```

- [ ] **Step 2: Create package structure**

Create these empty `__init__.py` files:

```
src/medmdt/__init__.py
src/medmdt/config/__init__.py
src/medmdt/extractor/__init__.py
src/medmdt/extractor/parsers/__init__.py
src/medmdt/knowledge/__init__.py
src/medmdt/llm/__init__.py
src/medmdt/llm/prompts/__init__.py
src/medmdt/llm/prompts/extraction/__init__.py
src/medmdt/llm/prompts/retrieval/__init__.py
tests/__init__.py
```

- [ ] **Step 3: Write the failing test for Settings**

```python
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
    os.environ["PADDLEOCR_TOKEN"] = "test"
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    del os.environ["PADDLEOCR_TOKEN"]
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement Settings**

```python
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

    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o"

    paddleocr_api_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    paddleocr_token: str
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
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Create experts.yaml with two starter experts**

```yaml
# config/experts.yaml
defaults:
  llm:
    provider: "openai"
    model: "gpt-4o"

experts:
  internist:
    name: "内科专家"
    knowledge_domains: ["internal_medicine"]
    system_prompt: |
      你是一位资深内科主任医师，拥有20年以上临床经验。
      你擅长内科常见病和疑难病的诊断与治疗。
      请基于提供的患者资料和医学知识，给出专业的分析和建议。
    llm:
      provider: "openai"
      model: "gpt-4o"

  radiologist:
    name: "影像科专家"
    knowledge_domains: ["radiology"]
    system_prompt: |
      你是一位资深影像科主任医师，擅长CT、MRI、X线等医学影像的判读与分析。
      请基于提供的影像描述和相关临床资料，给出专业的影像学分析和诊断建议。
    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
```

- [ ] **Step 7: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
      ETCD_SNAPSHOT_COUNT: "50000"
    volumes:
      - milvus_etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  milvus-minio:
    image: minio/minio:RELEASE.2024-09-22T00-33-43Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
    volumes:
      - milvus_minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.4-latest
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - milvus-etcd
      - milvus-minio
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    volumes:
      - milvus_data:/var/lib/milvus

  elasticsearch:
    image: elasticsearch:8.15.0
    ports:
      - "9200:9200"
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: "-Xms512m -Xmx512m"
    volumes:
      - es_data:/usr/share/elasticsearch/data

volumes:
  neo4j_data:
  milvus_etcd:
  milvus_minio:
  milvus_data:
  es_data:
```

- [ ] **Step 8: Create .env.example**

```env
# .env.example
MEDMDT_PADDLEOCR_TOKEN=your-paddleocr-token
MEDMDT_NEO4J_PASSWORD=password
MEDMDT_DEFAULT_LLM_PROVIDER=openai
MEDMDT_DEFAULT_LLM_MODEL=gpt-4o

# LLM API Keys (set the ones you use)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DASHSCOPE_API_KEY=sk-...
ZHIPUAI_API_KEY=...
MOONSHOT_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

- [ ] **Step 9: Create tests/conftest.py with shared fixtures**

```python
# tests/conftest.py
import os
import pytest
from medmdt.config.settings import Settings


@pytest.fixture
def settings():
    return Settings(paddleocr_token="test-token")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from medmdt.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example config/ src/medmdt/ tests/
git commit -m "feat: project foundation — config, dependencies, docker-compose"
```

---

### Task 2: LLM Provider Factory

**Files:**
- Create: `src/medmdt/llm/provider.py`
- Test: `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: `Settings` from Task 1 (`default_llm_provider`, `default_llm_model`)
- Produces: `create_chat_model(provider: str, model: str, **kwargs) -> BaseChatModel` — factory function used by every agent and prompt call. Also `LLMConfig` dataclass with fields `provider`, `model`, `base_url`, `api_key_env`, `temperature`, `max_tokens`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_provider.py
from unittest.mock import patch, MagicMock
import pytest
from medmdt.llm.provider import create_chat_model, LLMConfig, PROVIDER_REGISTRY


def test_llm_config_defaults():
    cfg = LLMConfig(provider="openai", model="gpt-4o")
    assert cfg.temperature == 0.0
    assert cfg.max_tokens is None


def test_provider_registry_has_all_providers():
    expected = {"openai", "anthropic", "qwen", "zhipu", "moonshot", "deepseek", "ollama", "custom"}
    assert expected == set(PROVIDER_REGISTRY.keys())


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_openai_model(mock_cls):
    mock_cls.return_value = MagicMock()
    model = create_chat_model("openai", "gpt-4o")
    mock_cls.assert_called_once_with(model="gpt-4o", temperature=0.0)
    assert model is mock_cls.return_value


@patch("medmdt.llm.provider.ChatAnthropic")
def test_create_anthropic_model(mock_cls):
    mock_cls.return_value = MagicMock()
    model = create_chat_model("anthropic", "claude-sonnet-4-6")
    mock_cls.assert_called_once_with(model="claude-sonnet-4-6", temperature=0.0)
    assert model is mock_cls.return_value


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_qwen_uses_dashscope_base_url(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("qwen", "qwen-plus")
    call_kwargs = mock_cls.call_args[1]
    assert "dashscope" in call_kwargs["base_url"]
    assert call_kwargs["model"] == "qwen-plus"


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_deepseek(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("deepseek", "deepseek-chat")
    call_kwargs = mock_cls.call_args[1]
    assert "deepseek" in call_kwargs["base_url"]


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_custom_requires_base_url(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("custom", "my-model", base_url="https://my.api/v1", api_key="sk-test")
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["base_url"] == "https://my.api/v1"
    assert call_kwargs["api_key"] == "sk-test"


def test_create_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_chat_model("nonexistent", "model")
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_llm_provider.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement provider.py**

```python
# src/medmdt/llm/provider.py
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


_OPENAI_COMPAT_URLS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

_OPENAI_COMPAT_KEY_ENVS: dict[str, str] = {
    "qwen": "DASHSCOPE_API_KEY",
    "zhipu": "ZHIPUAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _create_openai(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _create_anthropic(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatAnthropic(model=model, temperature=temperature, **kwargs)


def _create_openai_compat(provider: str, model: str, temperature: float, **kwargs) -> BaseChatModel:
    base_url = kwargs.pop("base_url", _OPENAI_COMPAT_URLS[provider])
    api_key = kwargs.pop("api_key", None)
    params = {"model": model, "base_url": base_url, "temperature": temperature, **kwargs}
    if api_key:
        params["api_key"] = api_key
    else:
        env_var = _OPENAI_COMPAT_KEY_ENVS.get(provider)
        if env_var:
            import os
            key = os.environ.get(env_var)
            if key:
                params["api_key"] = key
    return ChatOpenAI(**params)


def _create_ollama(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOllama(model=model, temperature=temperature, **kwargs)


def _create_custom(model: str, temperature: float, **kwargs) -> BaseChatModel:
    base_url = kwargs.pop("base_url", None)
    api_key = kwargs.pop("api_key", None)
    if not base_url:
        raise ValueError("Custom provider requires base_url")
    params = {"model": model, "base_url": base_url, "temperature": temperature, **kwargs}
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**params)


PROVIDER_REGISTRY: dict[str, callable] = {
    "openai": lambda m, t, **kw: _create_openai(m, t, **kw),
    "anthropic": lambda m, t, **kw: _create_anthropic(m, t, **kw),
    "qwen": lambda m, t, **kw: _create_openai_compat("qwen", m, t, **kw),
    "zhipu": lambda m, t, **kw: _create_openai_compat("zhipu", m, t, **kw),
    "moonshot": lambda m, t, **kw: _create_openai_compat("moonshot", m, t, **kw),
    "deepseek": lambda m, t, **kw: _create_openai_compat("deepseek", m, t, **kw),
    "ollama": lambda m, t, **kw: _create_ollama(m, t, **kw),
    "custom": lambda m, t, **kw: _create_custom(m, t, **kw),
}


def create_chat_model(provider: str, model: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
    factory = PROVIDER_REGISTRY.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_REGISTRY.keys())}")
    return factory(model, temperature, **kwargs)
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_llm_provider.py -v`
Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/llm/ tests/test_llm_provider.py
git commit -m "feat: LLM provider factory — 8 providers via unified interface"
```

---

### Task 3: Extraction Schemas

**Files:**
- Create: `src/medmdt/extractor/schemas.py`
- Test: `tests/test_extraction_schemas.py`

**Interfaces:**
- Produces: Pydantic models used by parsers (Task 4/5) and ingestor (Task 8):
  - `SourceInfo(file: str, type: str, page: int | None)`
  - `Entity(name: str, type: str, aliases: list[str])`
  - `Relation(head: str, relation: str, tail: str, evidence: str, confidence: float)`
  - `TextChunk(text: str, summary: str, keywords: list[str], metadata: dict)`
  - `ExtractionResult(source: SourceInfo, entities: list[Entity], relations: list[Relation], chunks: list[TextChunk])`
  - `ParsedPage(page_num: int, markdown: str, images: list[bytes])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extraction_schemas.py
import pytest
from medmdt.extractor.schemas import (
    SourceInfo, Entity, Relation, TextChunk, ExtractionResult, ParsedPage,
)


def test_source_info():
    s = SourceInfo(file="test.pdf", type="clinical_guideline", page=1)
    assert s.file == "test.pdf"
    assert s.page == 1


def test_source_info_page_optional():
    s = SourceInfo(file="test.jpg", type="image")
    assert s.page is None


def test_entity():
    e = Entity(name="2型糖尿病", type="disease", aliases=["T2DM"])
    assert e.name == "2型糖尿病"
    assert "T2DM" in e.aliases


def test_entity_type_validated():
    valid_types = {"disease", "symptom", "drug", "procedure", "anatomy", "lab_test", "other"}
    for t in valid_types:
        Entity(name="test", type=t, aliases=[])
    with pytest.raises(Exception):
        Entity(name="test", type="invalid_type", aliases=[])


def test_relation():
    r = Relation(
        head="二甲双胍", relation="first_line_treatment_for",
        tail="2型糖尿病", evidence="指南原文...", confidence=0.95,
    )
    assert r.confidence == 0.95


def test_relation_confidence_range():
    with pytest.raises(Exception):
        Relation(head="a", relation="r", tail="b", evidence="e", confidence=1.5)


def test_text_chunk():
    c = TextChunk(
        text="原始文本", summary="摘要",
        keywords=["关键词"], metadata={"section": "治疗"},
    )
    assert c.keywords == ["关键词"]


def test_extraction_result():
    result = ExtractionResult(
        source=SourceInfo(file="test.pdf", type="clinical_guideline"),
        entities=[Entity(name="糖尿病", type="disease", aliases=[])],
        relations=[],
        chunks=[TextChunk(text="text", summary="sum", keywords=[], metadata={})],
    )
    assert len(result.entities) == 1
    assert len(result.chunks) == 1


def test_extraction_result_to_dict_roundtrip():
    result = ExtractionResult(
        source=SourceInfo(file="test.pdf", type="clinical_guideline"),
        entities=[Entity(name="糖尿病", type="disease", aliases=["DM"])],
        relations=[Relation(head="a", relation="r", tail="b", evidence="e", confidence=0.9)],
        chunks=[TextChunk(text="t", summary="s", keywords=["k"], metadata={"x": 1})],
    )
    d = result.model_dump()
    restored = ExtractionResult.model_validate(d)
    assert restored == result


def test_parsed_page():
    p = ParsedPage(page_num=0, markdown="# Title", images=[b"fake-png"])
    assert p.page_num == 0
    assert len(p.images) == 1
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_schemas.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement schemas.py**

```python
# src/medmdt/extractor/schemas.py
from pydantic import BaseModel, Field, field_validator


class SourceInfo(BaseModel):
    file: str
    type: str
    page: int | None = None


ENTITY_TYPES = {"disease", "symptom", "drug", "procedure", "anatomy", "lab_test", "other"}


class Entity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {v}. Must be one of {ENTITY_TYPES}")
        return v


class Relation(BaseModel):
    head: str
    relation: str
    tail: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class TextChunk(BaseModel):
    text: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    source: SourceInfo
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)


class ParsedPage(BaseModel):
    page_num: int
    markdown: str
    images: list[bytes] = Field(default_factory=list)
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_schemas.py -v`
Expected: 10 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/extractor/schemas.py tests/test_extraction_schemas.py
git commit -m "feat: extraction schemas — Entity, Relation, TextChunk, ExtractionResult"
```

---

### Task 4: PDF Parser (PaddleOCR-VL-1.6)

**Files:**
- Create: `src/medmdt/extractor/parsers/pdf_parser.py`
- Test: `tests/test_pdf_parser.py`

**Interfaces:**
- Consumes: `Settings` (Task 1) — `paddleocr_api_url`, `paddleocr_token`, `paddleocr_use_*` flags; `ParsedPage` (Task 3)
- Produces: `PaddleOCRClient.parse(file_path: str) -> list[ParsedPage]` — submits PDF to PaddleOCR API, polls for completion, returns per-page markdown + images.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdf_parser.py
from unittest.mock import patch, MagicMock
import json
import pytest
from medmdt.config.settings import Settings
from medmdt.extractor.parsers.pdf_parser import PaddleOCRClient
from medmdt.extractor.schemas import ParsedPage


@pytest.fixture
def settings():
    return Settings(paddleocr_token="test-token")


@pytest.fixture
def client(settings):
    return PaddleOCRClient(settings)


def _mock_submit_response(job_id="job-123"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"jobId": job_id}}
    return resp


def _mock_done_response(result_url="https://example.com/result.jsonl"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "state": "done",
            "extractProgress": {
                "extractedPages": 1,
                "startTime": "2026-01-01T00:00:00",
                "endTime": "2026-01-01T00:01:00",
            },
            "resultUrl": {"jsonUrl": result_url},
        }
    }
    return resp


def _mock_jsonl_response():
    page_result = {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "text": "# 糖尿病诊疗指南\n\n二甲双胍是一线用药。",
                        "images": {},
                    },
                    "outputImages": {},
                }
            ]
        }
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(page_result)
    resp.raise_for_status = MagicMock()
    return resp


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_local_file(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    mock_requests.post.return_value = _mock_submit_response()
    mock_requests.get.side_effect = [_mock_done_response(), _mock_jsonl_response()]

    pages = client.parse(str(pdf_file))

    assert len(pages) == 1
    assert isinstance(pages[0], ParsedPage)
    assert "糖尿病" in pages[0].markdown
    assert pages[0].page_num == 0


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_url(mock_requests, client):
    mock_requests.post.return_value = _mock_submit_response()
    mock_requests.get.side_effect = [_mock_done_response(), _mock_jsonl_response()]

    pages = client.parse("https://example.com/doc.pdf")

    assert len(pages) == 1
    post_kwargs = mock_requests.post.call_args
    assert "json" in post_kwargs.kwargs or (len(post_kwargs.args) > 1)


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_handles_failed_job(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    mock_requests.post.return_value = _mock_submit_response()
    failed_resp = MagicMock()
    failed_resp.status_code = 200
    failed_resp.json.return_value = {
        "data": {"state": "failed", "errorMsg": "Unsupported format"}
    }
    mock_requests.get.return_value = failed_resp

    with pytest.raises(RuntimeError, match="Unsupported format"):
        client.parse(str(pdf_file))


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_with_embedded_images(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    mock_requests.post.return_value = _mock_submit_response()

    page_result = {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "text": "# Page with image\n\n![xray](img_0.png)",
                        "images": {"img_0.png": "https://example.com/img_0.png"},
                    },
                    "outputImages": {},
                }
            ]
        }
    }
    jsonl_resp = MagicMock()
    jsonl_resp.status_code = 200
    jsonl_resp.text = json.dumps(page_result)
    jsonl_resp.raise_for_status = MagicMock()

    img_resp = MagicMock()
    img_resp.status_code = 200
    img_resp.content = b"fake-png-bytes"

    mock_requests.get.side_effect = [_mock_done_response(), jsonl_resp, img_resp]

    pages = client.parse(str(pdf_file))

    assert len(pages) == 1
    assert len(pages[0].images) == 1
    assert pages[0].images[0] == b"fake-png-bytes"
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_pdf_parser.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement pdf_parser.py**

```python
# src/medmdt/extractor/parsers/pdf_parser.py
import json
import time
import requests
from pathlib import Path

from medmdt.config.settings import Settings
from medmdt.extractor.schemas import ParsedPage


class PaddleOCRClient:
    def __init__(self, settings: Settings):
        self._api_url = settings.paddleocr_api_url
        self._headers = {"Authorization": f"bearer {settings.paddleocr_token}"}
        self._optional_payload = {
            "useDocOrientationClassify": settings.paddleocr_use_doc_orientation_classify,
            "useDocUnwarping": settings.paddleocr_use_doc_unwarping,
            "useChartRecognition": settings.paddleocr_use_chart_recognition,
        }

    def parse(self, file_path: str, poll_interval: float = 5.0) -> list[ParsedPage]:
        job_id = self._submit(file_path)
        jsonl_url = self._poll_until_done(job_id, poll_interval)
        return self._fetch_results(jsonl_url)

    def _submit(self, file_path: str) -> str:
        if file_path.startswith("http"):
            headers = {**self._headers, "Content-Type": "application/json"}
            payload = {
                "fileUrl": file_path,
                "model": "PaddleOCR-VL-1.6",
                "optionalPayload": self._optional_payload,
            }
            resp = requests.post(self._api_url, json=payload, headers=headers)
        else:
            data = {
                "model": "PaddleOCR-VL-1.6",
                "optionalPayload": json.dumps(self._optional_payload),
            }
            with open(file_path, "rb") as f:
                resp = requests.post(
                    self._api_url, headers=self._headers, data=data, files={"file": f},
                )

        resp.raise_for_status()
        return resp.json()["data"]["jobId"]

    def _poll_until_done(self, job_id: str, poll_interval: float) -> str:
        while True:
            resp = requests.get(f"{self._api_url}/{job_id}", headers=self._headers)
            resp.raise_for_status()
            data = resp.json()["data"]
            state = data["state"]

            if state == "done":
                return data["resultUrl"]["jsonUrl"]
            elif state == "failed":
                raise RuntimeError(f"PaddleOCR job failed: {data.get('errorMsg', 'unknown')}")

            time.sleep(poll_interval)

    def _fetch_results(self, jsonl_url: str) -> list[ParsedPage]:
        resp = requests.get(jsonl_url)
        resp.raise_for_status()

        pages: list[ParsedPage] = []
        page_num = 0

        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)["result"]

            for layout in result["layoutParsingResults"]:
                md_text = layout["markdown"]["text"]
                md_images = layout["markdown"].get("images", {})

                image_bytes_list: list[bytes] = []
                for _img_name, img_url in md_images.items():
                    img_resp = requests.get(img_url)
                    if img_resp.status_code == 200:
                        image_bytes_list.append(img_resp.content)

                pages.append(ParsedPage(
                    page_num=page_num,
                    markdown=md_text,
                    images=image_bytes_list,
                ))
                page_num += 1

        return pages
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_pdf_parser.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/extractor/parsers/pdf_parser.py tests/test_pdf_parser.py
git commit -m "feat: PDF parser — PaddleOCR-VL-1.6 API integration"
```

---

### Task 5: Image Parser (Multimodal LLM)

**Files:**
- Create: `src/medmdt/extractor/parsers/image_parser.py`
- Test: `tests/test_image_parser.py`

**Interfaces:**
- Consumes: `create_chat_model(provider, model)` from Task 2
- Produces: `ImageParser.analyze(image_data: bytes, context: str = "") -> ImageAnalysisResult` where `ImageAnalysisResult(description: str, findings: list[str], modality: str | None)`. Also `ImageParser.analyze_batch(images: list[bytes], context: str = "") -> list[ImageAnalysisResult]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_parser.py
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from medmdt.extractor.parsers.image_parser import ImageParser, ImageAnalysisResult


def test_image_analysis_result():
    r = ImageAnalysisResult(
        description="右肺中叶见斑片状高密度影",
        findings=["右肺中叶炎症", "心影大小正常"],
        modality="X-ray",
    )
    assert len(r.findings) == 2


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
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_image_parser.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement image_parser.py**

```python
# src/medmdt/extractor/parsers/image_parser.py
import base64
import json
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

ANALYSIS_PROMPT = """你是一位资深医学影像分析专家。请分析这张医学图像，输出JSON格式：
{
  "description": "影像的整体描述",
  "findings": ["发现1", "发现2", ...],
  "modality": "影像类型，如X-ray/CT/MRI/病理切片/超声等，无法判断则为null"
}

只输出JSON，不要其他文字。"""


class ImageAnalysisResult(BaseModel):
    description: str
    findings: list[str] = Field(default_factory=list)
    modality: str | None = None


class ImageParser:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def analyze(self, image_data: bytes, context: str = "") -> ImageAnalysisResult:
        b64 = base64.b64encode(image_data).decode()
        content = [
            {"type": "text", "text": self._build_prompt(context)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        message = HumanMessage(content=content)
        response = self._llm.invoke([message])
        return ImageAnalysisResult.model_validate_json(response.content)

    def analyze_batch(self, images: list[bytes], context: str = "") -> list[ImageAnalysisResult]:
        return [self.analyze(img, context) for img in images]

    def _build_prompt(self, context: str) -> str:
        if context:
            return f"{ANALYSIS_PROMPT}\n\n临床背景：{context}"
        return ANALYSIS_PROMPT
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_image_parser.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/extractor/parsers/image_parser.py tests/test_image_parser.py
git commit -m "feat: image parser — multimodal LLM medical image analysis"
```

---

### Task 6: Neo4j Graph Store

**Files:**
- Create: `src/medmdt/knowledge/graph_store.py`
- Test: `tests/test_graph_store.py`

**Interfaces:**
- Consumes: `Settings` (Task 1) — `neo4j_uri`, `neo4j_user`, `neo4j_password`; `Entity`, `Relation` from Task 3
- Produces:
  - `GraphStore.__init__(uri, user, password)`
  - `GraphStore.upsert_entities(entities: list[Entity]) -> int` — returns count of created/merged nodes
  - `GraphStore.upsert_relations(relations: list[Relation]) -> int` — returns count of created/merged relationships
  - `GraphStore.query_by_entity(name: str) -> list[dict]` — returns related entities and relationships
  - `GraphStore.query_differential(symptoms: list[str]) -> list[dict]` — returns diseases matching symptoms ranked by match count
  - `GraphStore.close()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_store.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.graph_store import GraphStore
from medmdt.extractor.schemas import Entity, Relation


@pytest.fixture
def mock_driver():
    with patch("medmdt.knowledge.graph_store.GraphDatabase") as mock_gdb:
        driver = MagicMock()
        mock_gdb.driver.return_value = driver
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        yield driver, session


def test_graph_store_init(mock_driver):
    driver, _ = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    assert store._driver is driver


def test_upsert_entities(mock_driver):
    driver, session = mock_driver
    session.run.return_value.consume.return_value.counters.nodes_created = 2

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    entities = [
        Entity(name="糖尿病", type="disease", aliases=["DM"]),
        Entity(name="二甲双胍", type="drug", aliases=[]),
    ]
    count = store.upsert_entities(entities)

    assert session.run.called
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher


def test_upsert_relations(mock_driver):
    driver, session = mock_driver
    session.run.return_value.consume.return_value.counters.relationships_created = 1

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    relations = [
        Relation(
            head="二甲双胍", relation="treats", tail="糖尿病",
            evidence="指南推荐", confidence=0.95,
        ),
    ]
    count = store.upsert_relations(relations)

    assert session.run.called


def test_query_by_entity(mock_driver):
    driver, session = mock_driver
    session.run.return_value.data.return_value = [
        {"related": "二甲双胍", "rel_type": "treated_by", "direction": "outgoing"}
    ]

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_by_entity("糖尿病")

    assert len(results) == 1
    assert results[0]["related"] == "二甲双胍"


def test_query_differential(mock_driver):
    driver, session = mock_driver
    session.run.return_value.data.return_value = [
        {"disease": "高血压", "match_count": 3},
        {"disease": "糖尿病", "match_count": 1},
    ]

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_differential(["头痛", "视物模糊", "血压升高"])

    assert len(results) == 2
    assert results[0]["disease"] == "高血压"


def test_close(mock_driver):
    driver, _ = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    store.close()
    driver.close.assert_called_once()
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_graph_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement graph_store.py**

```python
# src/medmdt/knowledge/graph_store.py
from neo4j import GraphDatabase
from medmdt.extractor.schemas import Entity, Relation


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def upsert_entities(self, entities: list[Entity]) -> int:
        total = 0
        with self._driver.session() as session:
            for entity in entities:
                result = session.run(
                    "MERGE (n:MedicalEntity {name: $name}) "
                    "SET n.type = $type, n.aliases = $aliases "
                    "WITH n "
                    "CALL apoc.create.addLabels(n, [$label]) YIELD node "
                    "RETURN node",
                    name=entity.name,
                    type=entity.type,
                    aliases=entity.aliases,
                    label=entity.type.capitalize(),
                )
                total += 1
        return total

    def upsert_relations(self, relations: list[Relation]) -> int:
        total = 0
        with self._driver.session() as session:
            for rel in relations:
                session.run(
                    "MERGE (h:MedicalEntity {name: $head}) "
                    "MERGE (t:MedicalEntity {name: $tail}) "
                    "MERGE (h)-[r:RELATED {type: $rel_type}]->(t) "
                    "SET r.evidence = $evidence, r.confidence = $confidence",
                    head=rel.head,
                    tail=rel.tail,
                    rel_type=rel.relation,
                    evidence=rel.evidence,
                    confidence=rel.confidence,
                )
                total += 1
        return total

    def query_by_entity(self, name: str) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:MedicalEntity {name: $name})-[r]-(m:MedicalEntity) "
                "RETURN m.name AS related, r.type AS rel_type, "
                "CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction",
                name=name,
            )
            return result.data()

    def query_differential(self, symptoms: list[str]) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:MedicalEntity)-[r:RELATED {type: 'indicates'}]->(d:MedicalEntity {type: 'disease'}) "
                "WHERE s.name IN $symptoms "
                "RETURN d.name AS disease, count(s) AS match_count "
                "ORDER BY match_count DESC",
                symptoms=symptoms,
            )
            return result.data()

    def close(self):
        self._driver.close()
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_graph_store.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/knowledge/graph_store.py tests/test_graph_store.py
git commit -m "feat: Neo4j graph store — entity/relation CRUD and differential query"
```

---

### Task 7: Milvus Vector Store

**Files:**
- Create: `src/medmdt/knowledge/vector_store.py`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `Settings` (Task 1) — `milvus_host`, `milvus_port`, `embedding_model`, `embedding_dim`
- Produces:
  - `VectorStore.__init__(host, port, collection_name, embedding_dim)`
  - `VectorStore.insert(texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> list[int]` — returns inserted IDs
  - `VectorStore.search(query_embedding: list[float], top_k: int = 10, domain: str | None = None) -> list[VectorSearchResult]` where `VectorSearchResult(text: str, score: float, metadata: dict)`
  - `VectorStore.ensure_collection()`
  - `VectorStore.close()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.vector_store import VectorStore, VectorSearchResult


def test_vector_search_result():
    r = VectorSearchResult(text="test", score=0.95, metadata={"domain": "guideline"})
    assert r.score == 0.95


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_vector_store_ensure_collection_creates_if_missing(
    mock_util, mock_coll_cls, mock_conn
):
    mock_util.has_collection.return_value = False
    store = VectorStore("localhost", 19530, "test_collection", 1024)
    store.ensure_collection()
    mock_util.has_collection.assert_called_with("test_collection")


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_insert(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll
    mock_coll.insert.return_value = MagicMock(primary_keys=[1, 2])

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    ids = store.insert(
        texts=["text1", "text2"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        metadatas=[{"domain": "guideline"}, {"domain": "case"}],
    )

    assert ids == [1, 2]
    mock_coll.insert.assert_called_once()


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_search(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll

    hit = MagicMock()
    hit.distance = 0.95
    hit.entity.get.side_effect = lambda k: {"text": "结果文本", "metadata": '{"domain": "guideline"}'}.get(k)
    mock_coll.search.return_value = [[hit]]

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    results = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], VectorSearchResult)
    assert results[0].score == 0.95
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_vector_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement vector_store.py**

```python
# src/medmdt/knowledge/vector_store.py
import json
from dataclasses import dataclass
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility,
)


@dataclass
class VectorSearchResult:
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, host: str, port: int, collection_name: str, embedding_dim: int):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._collection: Collection | None = None
        connections.connect(alias="default", host=host, port=port)

    def ensure_collection(self):
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields)
        self._collection = Collection(self._collection_name, schema)

        from pymilvus import IndexType
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index("embedding", index_params)
        self._collection.load()

    def insert(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        data = [
            texts,
            embeddings,
            [json.dumps(m, ensure_ascii=False) for m in metadatas],
        ]
        result = self._collection.insert(data)
        self._collection.flush()
        return result.primary_keys

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        domain: str | None = None,
    ) -> list[VectorSearchResult]:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        expr = None
        if domain:
            expr = f'metadata like "%\\"domain\\": \\"{domain}\\"%"'

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        search_results = []
        for hit in results[0]:
            meta_str = hit.entity.get("metadata")
            meta = json.loads(meta_str) if meta_str else {}
            search_results.append(VectorSearchResult(
                text=hit.entity.get("text"),
                score=hit.distance,
                metadata=meta,
            ))
        return search_results

    def close(self):
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_vector_store.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/knowledge/vector_store.py tests/test_vector_store.py
git commit -m "feat: Milvus vector store — insert, cosine search, domain filtering"
```

---

### Task 8: Elasticsearch Keyword Store

**Files:**
- Create: `src/medmdt/knowledge/keyword_store.py`
- Test: `tests/test_keyword_store.py`

**Interfaces:**
- Consumes: `Settings` (Task 1) — `elasticsearch_url`
- Produces:
  - `KeywordStore.__init__(url, index_name)`
  - `KeywordStore.ensure_index()`
  - `KeywordStore.index_chunks(chunks: list[TextChunk], source: SourceInfo) -> int` — returns count of indexed docs
  - `KeywordStore.search(query: str, top_k: int = 10, filters: dict | None = None) -> list[KeywordSearchResult]` where `KeywordSearchResult(text: str, score: float, keywords: list[str], metadata: dict, source: dict)`
  - `KeywordStore.close()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keyword_store.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.keyword_store import KeywordStore, KeywordSearchResult
from medmdt.extractor.schemas import TextChunk, SourceInfo


def test_keyword_search_result():
    r = KeywordSearchResult(
        text="test", score=5.0, keywords=["k1"],
        metadata={"section": "治疗"}, source={"file": "a.pdf"},
    )
    assert r.score == 5.0


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_keyword_store_init(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    assert store._es is mock_es


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_ensure_index_creates_if_missing(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.indices.exists.return_value = False

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.ensure_index()

    mock_es.indices.create.assert_called_once()
    call_kwargs = mock_es.indices.create.call_args
    assert call_kwargs[1]["index"] == "medmdt_chunks" or call_kwargs[0][0] == "medmdt_chunks"


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_index_chunks(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    chunks = [
        TextChunk(text="糖尿病治疗", summary="治疗方案", keywords=["糖尿病"], metadata={"section": "治疗"}),
        TextChunk(text="高血压治疗", summary="治疗方案", keywords=["高血压"], metadata={"section": "治疗"}),
    ]
    source = SourceInfo(file="guide.pdf", type="clinical_guideline", page=1)
    count = store.index_chunks(chunks, source)

    assert count == 2
    assert mock_es.index.call_count == 2


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_search(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 5.5,
                    "_source": {
                        "text": "糖尿病诊断标准",
                        "keywords": ["糖尿病", "诊断"],
                        "metadata": {"section": "诊断"},
                        "source": {"file": "guide.pdf", "type": "clinical_guideline"},
                    },
                }
            ]
        }
    }

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    results = store.search("糖尿病诊断", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], KeywordSearchResult)
    assert results[0].score == 5.5
    assert "糖尿病" in results[0].keywords


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_search_with_filters(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.search.return_value = {"hits": {"hits": []}}

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.search("test", filters={"source.type": "clinical_guideline"})

    search_body = mock_es.search.call_args[1]["body"]
    assert "bool" in search_body["query"]
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_keyword_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement keyword_store.py**

```python
# src/medmdt/knowledge/keyword_store.py
from dataclasses import dataclass, field
from elasticsearch import Elasticsearch
from medmdt.extractor.schemas import TextChunk, SourceInfo


@dataclass
class KeywordSearchResult:
    text: str
    score: float
    keywords: list[str]
    metadata: dict
    source: dict


INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "ik_smart_analyzer"},
            "summary": {"type": "text", "analyzer": "ik_smart_analyzer"},
            "keywords": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "source": {
                "properties": {
                    "file": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
        }
    },
}

INDEX_SETTINGS_FALLBACK = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "summary": {"type": "text"},
            "keywords": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "source": {
                "properties": {
                    "file": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
        }
    },
}


class KeywordStore:
    def __init__(self, url: str, index_name: str = "medmdt_chunks"):
        self._es = Elasticsearch(url)
        self._index_name = index_name

    def ensure_index(self):
        if self._es.indices.exists(index=self._index_name):
            return
        try:
            self._es.indices.create(index=self._index_name, body=INDEX_SETTINGS)
        except Exception:
            self._es.indices.create(index=self._index_name, body=INDEX_SETTINGS_FALLBACK)

    def index_chunks(self, chunks: list[TextChunk], source: SourceInfo) -> int:
        count = 0
        source_dict = source.model_dump()
        for chunk in chunks:
            doc = {
                "text": chunk.text,
                "summary": chunk.summary,
                "keywords": chunk.keywords,
                "metadata": chunk.metadata,
                "source": source_dict,
            }
            self._es.index(index=self._index_name, body=doc)
            count += 1
        return count

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[KeywordSearchResult]:
        if filters:
            filter_clauses = [
                {"term": {k: v}} for k, v in filters.items()
            ]
            body = {
                "query": {
                    "bool": {
                        "must": {"multi_match": {"query": query, "fields": ["text", "summary", "keywords"]}},
                        "filter": filter_clauses,
                    }
                },
                "size": top_k,
            }
        else:
            body = {
                "query": {
                    "multi_match": {"query": query, "fields": ["text", "summary", "keywords"]}
                },
                "size": top_k,
            }

        resp = self._es.search(index=self._index_name, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(KeywordSearchResult(
                text=src["text"],
                score=hit["_score"],
                keywords=src.get("keywords", []),
                metadata=src.get("metadata", {}),
                source=src.get("source", {}),
            ))
        return results

    def close(self):
        self._es.close()
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_keyword_store.py -v`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/knowledge/keyword_store.py tests/test_keyword_store.py
git commit -m "feat: Elasticsearch keyword store — BM25 indexing and search"
```

---

### Task 9: Ingestor — Three-Store Writer

**Files:**
- Create: `src/medmdt/extractor/ingestor.py`
- Test: `tests/test_ingestor.py`

**Interfaces:**
- Consumes: `GraphStore` (Task 6), `VectorStore` (Task 7), `KeywordStore` (Task 8), `ExtractionResult` (Task 3). Also needs an embedding function — takes `create_chat_model` (Task 2) output or a dedicated embedding model.
- Produces: `Ingestor.__init__(graph_store, vector_store, keyword_store, embed_fn)` and `Ingestor.ingest(result: ExtractionResult) -> IngestReport` where `IngestReport(entities_count: int, relations_count: int, chunks_count: int)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestor.py
from unittest.mock import MagicMock
import pytest
from medmdt.extractor.ingestor import Ingestor, IngestReport
from medmdt.extractor.schemas import (
    ExtractionResult, SourceInfo, Entity, Relation, TextChunk,
)


@pytest.fixture
def mock_stores():
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 1024])
    return graph, vector, keyword, embed_fn


@pytest.fixture
def sample_result():
    return ExtractionResult(
        source=SourceInfo(file="guide.pdf", type="clinical_guideline", page=1),
        entities=[
            Entity(name="糖尿病", type="disease", aliases=["DM"]),
            Entity(name="二甲双胍", type="drug", aliases=[]),
        ],
        relations=[
            Relation(head="二甲双胍", relation="treats", tail="糖尿病", evidence="原文", confidence=0.9),
        ],
        chunks=[
            TextChunk(text="糖尿病治疗方案", summary="治疗", keywords=["糖尿病"], metadata={}),
        ],
    )


def test_ingest_report():
    r = IngestReport(entities_count=2, relations_count=1, chunks_count=3)
    assert r.entities_count == 2


def test_ingest_writes_to_all_stores(mock_stores, sample_result):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    report = ingestor.ingest(sample_result)

    assert isinstance(report, IngestReport)
    graph.upsert_entities.assert_called_once_with(sample_result.entities)
    graph.upsert_relations.assert_called_once_with(sample_result.relations)
    vector.insert.assert_called_once()
    keyword.index_chunks.assert_called_once()


def test_ingest_embeds_chunk_texts(mock_stores, sample_result):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    ingestor.ingest(sample_result)

    embed_fn.assert_called_once_with(["糖尿病治疗方案"])


def test_ingest_empty_result(mock_stores):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 0
    graph.upsert_relations.return_value = 0

    result = ExtractionResult(
        source=SourceInfo(file="empty.pdf", type="other"),
        entities=[], relations=[], chunks=[],
    )
    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    report = ingestor.ingest(result)

    assert report.entities_count == 0
    assert report.chunks_count == 0
    vector.insert.assert_not_called()
    keyword.index_chunks.assert_not_called()
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_ingestor.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement ingestor.py**

```python
# src/medmdt/extractor/ingestor.py
from dataclasses import dataclass
from typing import Callable

from medmdt.extractor.schemas import ExtractionResult
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore


@dataclass
class IngestReport:
    entities_count: int
    relations_count: int
    chunks_count: int


class Ingestor:
    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ):
        self._graph = graph_store
        self._vector = vector_store
        self._keyword = keyword_store
        self._embed_fn = embed_fn

    def ingest(self, result: ExtractionResult) -> IngestReport:
        entities_count = self._graph.upsert_entities(result.entities)
        relations_count = self._graph.upsert_relations(result.relations)

        chunks_count = 0
        if result.chunks:
            texts = [c.text for c in result.chunks]
            embeddings = self._embed_fn(texts)
            metadatas = [
                {
                    "summary": c.summary,
                    "keywords": c.keywords,
                    "domain": result.source.type,
                    **c.metadata,
                }
                for c in result.chunks
            ]
            self._vector.insert(texts, embeddings, metadatas)
            self._keyword.index_chunks(result.chunks, result.source)
            chunks_count = len(result.chunks)

        return IngestReport(
            entities_count=entities_count,
            relations_count=relations_count,
            chunks_count=chunks_count,
        )
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_ingestor.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/extractor/ingestor.py tests/test_ingestor.py
git commit -m "feat: ingestor — writes extraction results to Neo4j, Milvus, Elasticsearch"
```

---

### Task 10: Fusion Retriever (RRF)

**Files:**
- Create: `src/medmdt/knowledge/schemas.py`
- Create: `src/medmdt/knowledge/retriever.py`
- Create: `src/medmdt/llm/prompts/retrieval/query_analysis.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `GraphStore` (Task 6), `VectorStore` (Task 7), `KeywordStore` (Task 8), `create_chat_model` (Task 2)
- Produces:
  - `RetrievalResult(text: str, score: float, source: str, metadata: dict)`
  - `FusionRetriever.__init__(graph_store, vector_store, keyword_store, llm, embed_fn)`
  - `FusionRetriever.retrieve(query: str, top_k: int = 10) -> list[RetrievalResult]`
  - `rrf_fuse(ranked_lists: list[list], k: int = 60, weights: list[float] | None = None) -> list[tuple[any, float]]` — standalone RRF function

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.retriever import FusionRetriever, rrf_fuse
from medmdt.knowledge.schemas import RetrievalResult


def test_retrieval_result():
    r = RetrievalResult(text="test", score=0.9, source="neo4j", metadata={})
    assert r.source == "neo4j"


def test_rrf_fuse_single_list():
    items = ["a", "b", "c"]
    fused = rrf_fuse([items])
    assert fused[0][0] == "a"
    assert fused[0][1] > fused[1][1]


def test_rrf_fuse_multiple_lists():
    list1 = ["a", "b", "c"]
    list2 = ["c", "a", "b"]
    fused = rrf_fuse([list1, list2])
    scores = {item: score for item, score in fused}
    assert scores["a"] > scores["c"]  # a is rank 1 in list1, rank 2 in list2


def test_rrf_fuse_with_weights():
    list1 = ["a", "b"]
    list2 = ["b", "a"]
    fused_equal = rrf_fuse([list1, list2], weights=[1.0, 1.0])
    fused_weighted = rrf_fuse([list1, list2], weights=[3.0, 1.0])
    scores_equal = {item: score for item, score in fused_equal}
    scores_weighted = {item: score for item, score in fused_weighted}
    assert scores_weighted["a"] > scores_weighted["b"]


def test_rrf_fuse_empty():
    result = rrf_fuse([])
    assert result == []


def test_fusion_retriever_retrieve():
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    llm = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 1024])

    llm_response = MagicMock()
    llm_response.content = '{"keywords": ["糖尿病"], "entities": ["糖尿病"], "weights": {"graph": 0.3, "vector": 0.4, "keyword": 0.3}}'
    llm.invoke.return_value = llm_response

    graph.query_by_entity.return_value = [
        {"related": "二甲双胍", "rel_type": "treats", "direction": "outgoing"}
    ]

    from medmdt.knowledge.vector_store import VectorSearchResult
    vector.search.return_value = [
        VectorSearchResult(text="向量结果1", score=0.9, metadata={}),
    ]

    from medmdt.knowledge.keyword_store import KeywordSearchResult
    keyword.search.return_value = [
        KeywordSearchResult(text="关键词结果1", score=5.0, keywords=["糖尿病"], metadata={}, source={}),
    ]

    retriever = FusionRetriever(graph, vector, keyword, llm, embed_fn)
    results = retriever.retrieve("糖尿病的一线治疗方案")

    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    llm.invoke.assert_called_once()
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_retriever.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Create knowledge/schemas.py**

```python
# src/medmdt/knowledge/schemas.py
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    text: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)
```

- [ ] **Step 3: Create query_analysis prompt**

```python
# src/medmdt/llm/prompts/retrieval/query_analysis.py
QUERY_ANALYSIS_PROMPT = """你是一个医学查询分析器。分析用户的查询意图，提取关键信息。

输出JSON格式：
{
  "keywords": ["关键词1", "关键词2"],
  "entities": ["医学实体1", "医学实体2"],
  "weights": {
    "graph": 0.33,
    "vector": 0.34,
    "keyword": 0.33
  }
}

权重说明：
- graph: 适合实体关系查询（如"某药物的禁忌症"、"某疾病的治疗方案"）
- vector: 适合语义检索（如"如何处理老年患者的低血糖"）
- keyword: 适合精确术语匹配（如"HbA1c正常范围"）

三个权重之和必须为1.0。只输出JSON。

用户查询：{query}"""
```

- [ ] **Step 4: Implement retriever.py**

```python
# src/medmdt/knowledge/retriever.py
import json
from typing import Callable
from collections import defaultdict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.knowledge.schemas import RetrievalResult
from medmdt.llm.prompts.retrieval.query_analysis import QUERY_ANALYSIS_PROMPT


def rrf_fuse(
    ranked_lists: list[list],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple]:
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = defaultdict(float)
    item_map: dict[str, any] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list):
            key = str(item)
            item_map[key] = item
            scores[key] += weight / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(item_map[key], score) for key, score in sorted_items]


class FusionRetriever:
    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        llm: BaseChatModel,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ):
        self._graph = graph_store
        self._vector = vector_store
        self._keyword = keyword_store
        self._llm = llm
        self._embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        analysis = self._analyze_query(query)
        keywords = analysis.get("keywords", [])
        entities = analysis.get("entities", [])
        weights_cfg = analysis.get("weights", {"graph": 0.33, "vector": 0.34, "keyword": 0.33})

        graph_results = self._search_graph(entities)
        vector_results = self._search_vector(query)
        keyword_results = self._search_keyword(query, keywords)

        weights = [weights_cfg["graph"], weights_cfg["vector"], weights_cfg["keyword"]]
        fused = rrf_fuse([graph_results, vector_results, keyword_results], weights=weights)

        results = []
        for item, score in fused[:top_k]:
            if isinstance(item, RetrievalResult):
                item.score = score
                results.append(item)
            else:
                results.append(RetrievalResult(text=str(item), score=score, source="fused"))
        return results

    def _analyze_query(self, query: str) -> dict:
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)

    def _search_graph(self, entities: list[str]) -> list[RetrievalResult]:
        results = []
        for entity in entities:
            relations = self._graph.query_by_entity(entity)
            for rel in relations:
                text = f"{entity} --[{rel.get('rel_type', '')}]--> {rel.get('related', '')}"
                results.append(RetrievalResult(
                    text=text, score=0.0, source="graph",
                    metadata={"entity": entity, "relation": rel},
                ))
        return results

    def _search_vector(self, query: str) -> list[RetrievalResult]:
        embeddings = self._embed_fn([query])
        hits = self._vector.search(embeddings[0])
        return [
            RetrievalResult(text=h.text, score=h.score, source="vector", metadata=h.metadata)
            for h in hits
        ]

    def _search_keyword(self, query: str, keywords: list[str]) -> list[RetrievalResult]:
        search_query = f"{query} {' '.join(keywords)}" if keywords else query
        hits = self._keyword.search(search_query)
        return [
            RetrievalResult(
                text=h.text, score=h.score, source="keyword",
                metadata={**h.metadata, "keywords": h.keywords},
            )
            for h in hits
        ]
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_retriever.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/medmdt/knowledge/schemas.py src/medmdt/knowledge/retriever.py src/medmdt/llm/prompts/retrieval/ tests/test_retriever.py
git commit -m "feat: fusion retriever — RRF-based triple-store search with query analysis"
```

---

### Task 11: Extraction Prompts + Extraction Agent

**Files:**
- Create: `src/medmdt/llm/prompts/extraction/entity_extraction.py`
- Create: `src/medmdt/llm/prompts/extraction/relation_extraction.py`
- Create: `src/medmdt/extractor/agent.py`
- Test: `tests/test_extraction_agent.py`

**Interfaces:**
- Consumes: `create_chat_model` (Task 2), `PaddleOCRClient` (Task 4), `ImageParser` (Task 5), `Ingestor` (Task 9), `ExtractionResult` / `ParsedPage` (Task 3)
- Produces: `ExtractionAgent.__init__(settings, graph_store, vector_store, keyword_store, embed_fn)` and `ExtractionAgent.process_file(file_path: str) -> list[IngestReport]` — end-to-end pipeline: file → parse → extract → ingest.

- [ ] **Step 1: Create entity extraction prompt**

```python
# src/medmdt/llm/prompts/extraction/entity_extraction.py
ENTITY_RELATION_PROMPT = """你是一个医学知识提取专家。从以下医学文本中提取实体和关系。

实体类型：disease（疾病）, symptom（症状）, drug（药物）, procedure（操作/手术）, anatomy（解剖部位）, lab_test（检验项目）, other（其他）

关系类型：treats（治疗）, indicates（提示/指向）, contraindicated_for（禁忌于）, located_in（位于）, diagnoses（诊断）, causes（导致）, first_line_treatment_for（一线治疗）, side_effect_of（副作用）

输出JSON格式：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "aliases": ["别名1"]}}
  ],
  "relations": [
    {{
      "head": "头实体名称",
      "relation": "关系类型",
      "tail": "尾实体名称",
      "evidence": "原文中支持该关系的证据文本",
      "confidence": 0.95
    }}
  ]
}}

注意：
1. 实体名称使用标准医学术语
2. confidence 范围 0.0-1.0，反映关系的确定程度
3. evidence 必须是原文中的原始文本片段
4. 只输出JSON

医学文本：
{text}"""
```

- [ ] **Step 2: Create chunking/summary prompt**

```python
# src/medmdt/llm/prompts/extraction/relation_extraction.py
CHUNK_SUMMARY_PROMPT = """你是一个医学文本分析专家。将以下文本拆分为知识块，并为每个块生成摘要和关键词。

输出JSON格式：
{{
  "chunks": [
    {{
      "text": "原始文本段落（保持原文）",
      "summary": "一句话摘要",
      "keywords": ["关键词1", "关键词2"],
      "metadata": {{
        "section": "所属章节（如：诊断标准、治疗方案、预后评估等）",
        "guideline_level": "推荐等级（如有，如：A级推荐、专家共识等）"
      }}
    }}
  ]
}}

注意：
1. 每个chunk应该是一个语义完整的段落
2. keywords提取3-5个核心术语
3. 只输出JSON

医学文本：
{text}"""
```

- [ ] **Step 3: Write the failing test for extraction agent**

```python
# tests/test_extraction_agent.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.schemas import ParsedPage
from medmdt.extractor.ingestor import IngestReport
from medmdt.config.settings import Settings


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
        ParsedPage(page_num=0, markdown="# 胸部X光报告", images=[b"fake-xray"]),
    ]
    mock_ocr_cls.return_value = mock_ocr

    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=llm,
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
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_agent.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement agent.py**

```python
# src/medmdt/extractor/agent.py
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from medmdt.config.settings import Settings
from medmdt.extractor.schemas import (
    ExtractionResult, SourceInfo, Entity, Relation, TextChunk, ParsedPage,
)
from medmdt.extractor.parsers.pdf_parser import PaddleOCRClient
from medmdt.extractor.parsers.image_parser import ImageParser
from medmdt.extractor.ingestor import Ingestor, IngestReport
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.llm.prompts.extraction.entity_extraction import ENTITY_RELATION_PROMPT
from medmdt.llm.prompts.extraction.relation_extraction import CHUNK_SUMMARY_PROMPT

from typing import Callable


class ExtractionAgent:
    def __init__(
        self,
        settings: Settings,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        llm: BaseChatModel,
    ):
        self._settings = settings
        self._llm = llm
        self._image_parser = ImageParser(llm=llm)
        self._ingestor = Ingestor(graph_store, vector_store, keyword_store, embed_fn)
        self._ocr_client = PaddleOCRClient(settings)

    def process_file(self, file_path: str) -> list[IngestReport]:
        file_type = self._detect_file_type(file_path)

        if file_type == "pdf":
            return self._process_pdf(file_path)
        elif file_type == "image":
            return [self._process_image(file_path)]
        elif file_type == "dicom":
            raise NotImplementedError("DICOM processing is planned for P4")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _process_pdf(self, file_path: str) -> list[IngestReport]:
        pages = self._ocr_client.parse(file_path)
        reports = []

        for page in pages:
            image_descriptions = []
            if page.images:
                for img_bytes in page.images:
                    analysis = self._image_parser.analyze(img_bytes)
                    image_descriptions.append(analysis.description)

            full_text = page.markdown
            if image_descriptions:
                full_text += "\n\n[嵌入图像分析]\n" + "\n".join(image_descriptions)

            result = self._extract_from_text(
                text=full_text,
                source=SourceInfo(
                    file=file_path,
                    type=self._classify_document(full_text),
                    page=page.page_num,
                ),
            )
            report = self._ingestor.ingest(result)
            reports.append(report)

        return reports

    def _process_image(self, file_path: str) -> IngestReport:
        with open(file_path, "rb") as f:
            image_data = f.read()

        analysis = self._image_parser.analyze(image_data)
        result = ExtractionResult(
            source=SourceInfo(file=file_path, type="image"),
            entities=[],
            relations=[],
            chunks=[TextChunk(
                text=analysis.description,
                summary=analysis.description,
                keywords=analysis.findings,
                metadata={"modality": analysis.modality or "unknown"},
            )],
        )
        return self._ingestor.ingest(result)

    def _extract_from_text(self, text: str, source: SourceInfo) -> ExtractionResult:
        entity_prompt = ENTITY_RELATION_PROMPT.format(text=text)
        entity_response = self._llm.invoke([HumanMessage(content=entity_prompt)])
        entity_data = json.loads(entity_response.content)

        chunk_prompt = CHUNK_SUMMARY_PROMPT.format(text=text)
        chunk_response = self._llm.invoke([HumanMessage(content=chunk_prompt)])
        chunk_data = json.loads(chunk_response.content)

        entities = [Entity.model_validate(e) for e in entity_data.get("entities", [])]
        relations = [Relation.model_validate(r) for r in entity_data.get("relations", [])]
        chunks = [TextChunk.model_validate(c) for c in chunk_data.get("chunks", [])]

        return ExtractionResult(
            source=source,
            entities=entities,
            relations=relations,
            chunks=chunks,
        )

    def _classify_document(self, text: str) -> str:
        text_lower = text[:500].lower()
        if any(kw in text_lower for kw in ["指南", "guideline", "共识", "consensus"]):
            return "clinical_guideline"
        if any(kw in text_lower for kw in ["病例", "case", "入院", "出院"]):
            return "case_report"
        if any(kw in text_lower for kw in ["教材", "textbook", "章", "chapter"]):
            return "textbook"
        return "other"

    @staticmethod
    def _detect_file_type(file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        elif suffix in {".dcm", ".dicom"}:
            return "dicom"
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            return "image"
        else:
            raise ValueError(f"Unknown file extension: {suffix}")
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_agent.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/medmdt/extractor/agent.py src/medmdt/llm/prompts/extraction/ tests/test_extraction_agent.py
git commit -m "feat: extraction agent — end-to-end PDF→structured JSON→triple-store pipeline"
```

---

### Task 12: Ingest CLI Script + Integration Smoke Test

**Files:**
- Create: `scripts/ingest.py`
- Test: `tests/test_ingest_script.py`

**Interfaces:**
- Consumes: `ExtractionAgent` (Task 11), `Settings` (Task 1), `create_chat_model` (Task 2), all stores (Tasks 6-8)
- Produces: CLI entry point `python scripts/ingest.py <file_or_directory>` — processes files and reports results.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_script.py
from unittest.mock import patch, MagicMock
import pytest


def test_ingest_script_imports():
    from scripts.ingest import build_agent, run_ingest


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
    (tmp_path / "c.txt").write_text("not a pdf")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 1
    mock_report.relations_count = 0
    mock_report.chunks_count = 1
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    results = run_ingest(str(tmp_path))

    assert len(results) == 2
    assert mock_agent.process_file.call_count == 2
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_ingest_script.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement ingest.py**

```python
# scripts/ingest.py
import sys
from pathlib import Path
from medmdt.config.settings import get_settings
from medmdt.llm.provider import create_chat_model
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.ingestor import IngestReport

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}


def build_agent() -> ExtractionAgent:
    settings = get_settings()
    llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)

    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore(
        settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
    )
    vector_store.ensure_collection()
    keyword_store = KeywordStore(settings.elasticsearch_url)
    keyword_store.ensure_index()

    embedding_llm = create_chat_model(settings.default_llm_provider, settings.embedding_model)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        return embeddings.embed_documents(texts)

    return ExtractionAgent(
        settings=settings,
        graph_store=graph_store,
        vector_store=vector_store,
        keyword_store=keyword_store,
        embed_fn=embed_fn,
        llm=llm,
    )


def run_ingest(path: str) -> list[IngestReport]:
    agent = build_agent()
    target = Path(path)
    all_reports = []

    if target.is_file():
        reports = agent.process_file(str(target))
        all_reports.extend(reports)
        _print_reports(str(target), reports)
    elif target.is_dir():
        files = [f for f in target.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        for f in sorted(files):
            try:
                reports = agent.process_file(str(f))
                all_reports.extend(reports)
                _print_reports(str(f), reports)
            except Exception as e:
                print(f"[ERROR] {f}: {e}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    return all_reports


def _print_reports(file_path: str, reports: list[IngestReport]):
    total_e = sum(r.entities_count for r in reports)
    total_r = sum(r.relations_count for r in reports)
    total_c = sum(r.chunks_count for r in reports)
    print(f"[OK] {file_path}: {total_e} entities, {total_r} relations, {total_c} chunks")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file_or_directory>")
        sys.exit(1)
    run_ingest(sys.argv[1])
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && PYTHONPATH=. uv run pytest tests/test_ingest_script.py -v`
Expected: 3 passed

- [ ] **Step 4: Run all P1 tests**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/ -v`
Expected: All tests pass (approx 37 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest.py tests/test_ingest_script.py
git commit -m "feat: ingest CLI script — batch process files into knowledge base"
```
