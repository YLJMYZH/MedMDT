# MedMDT P3: API Gateway + Streaming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI API gateway that exposes the MDT consultation engine and knowledge pipeline as HTTP endpoints, with WebSocket streaming for real-time consultation progress.

**Architecture:** FastAPI application with two route groups: (1) consultation endpoints for creating/querying MDT sessions with WebSocket streaming, (2) knowledge management endpoints for ingesting files and searching the knowledge base. A dependency injection layer wires up stores, retriever, experts, and moderator. The consultation runs in a background task, with state tracked in an in-memory store keyed by consultation ID. WebSocket clients receive real-time events (round updates, expert opinions, final report).

**Tech Stack:** Python 3.12, FastAPI, uvicorn, python-multipart (file uploads), Pydantic v2, WebSocket, asyncio, pytest + httpx (async test client)

## Global Constraints

- Python >=3.12, managed by uv
- Source layout: `src/medmdt/` (src-layout)
- LLM access via `create_chat_model(provider, model)` from `medmdt.llm.provider`
- MDT graph via `build_mdt_graph(experts, moderator, retriever, selector_llm)` and `run_consultation(graph, patient_info, medical_records, max_rounds)` from `medmdt.mdt.graph`
- Knowledge retrieval via `FusionRetriever.retrieve(query, top_k)` from `medmdt.knowledge.retriever`
- Settings from `medmdt.config.settings.Settings` via `get_settings()`
- Expert creation via `create_all_experts(yaml_path)` from `medmdt.mdt.experts.factory`
- Tests: pytest with httpx AsyncClient; no real database or LLM calls in unit tests
- Add `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx` to pyproject.toml dependencies

---

### Task 1: API Request/Response Models + Dependencies Setup

**Files:**
- Create: `src/medmdt/api/__init__.py`
- Create: `src/medmdt/api/models.py`
- Create: `src/medmdt/api/deps.py`
- Modify: `pyproject.toml` (add fastapi, uvicorn, python-multipart, httpx)
- Test: `tests/test_api_models.py`

**Interfaces:**
- Consumes: `Settings` / `get_settings()` from P1; `create_chat_model` from P1; `create_all_experts` from P2; `Moderator` from P2; `FusionRetriever` from P1; `build_mdt_graph` from P2; `GraphStore`, `VectorStore`, `KeywordStore` from P1
- Produces:
  - Request models: `ConsultationRequest(patient_info: dict, medical_records: list[dict], max_rounds: int = 3)`, `KnowledgeSearchRequest(query: str, top_k: int = 10)`, `IngestRequest` (file upload metadata)
  - Response models: `ConsultationResponse(id: str, status: str, created_at: str)`, `ConsultationResult(id: str, status: str, final_report: str | None, discussion_rounds: list[dict], consensus: dict | None, divergences: list[str])`, `SearchResponse(results: list[dict], total: int)`, `IngestResponse(job_id: str, status: str, message: str)`
  - `ConsultationStatus` enum: `pending`, `running`, `completed`, `failed`
  - `ConsultationStore` — in-memory dict store: `create(req) -> str`, `get(id) -> dict | None`, `update(id, **fields)`, `list_all() -> list[dict]`
  - `get_consultation_store() -> ConsultationStore` — FastAPI dependency (singleton)

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.9`, `httpx>=0.27` to the dependencies list in `pyproject.toml`.

Run: `cd /Users/yljm/MedMDT && uv sync`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_api_models.py
import pytest
from medmdt.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    ConsultationResult,
    ConsultationStatus,
    KnowledgeSearchRequest,
    SearchResponse,
    IngestResponse,
    ConsultationStore,
)


def test_consultation_request():
    req = ConsultationRequest(
        patient_info={"chief_complaint": "头痛头晕1月", "age": 55, "gender": "male"},
        medical_records=[{"type": "vital", "content": "BP 160/100mmHg"}],
    )
    assert req.max_rounds == 3
    assert req.patient_info["chief_complaint"] == "头痛头晕1月"


def test_consultation_request_custom_rounds():
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        max_rounds=5,
    )
    assert req.max_rounds == 5


def test_consultation_response():
    resp = ConsultationResponse(
        id="c-123", status="pending", created_at="2026-07-02T10:00:00",
    )
    assert resp.id == "c-123"
    assert resp.status == "pending"


def test_consultation_result():
    result = ConsultationResult(
        id="c-123",
        status="completed",
        final_report="# 报告",
        discussion_rounds=[{"round_num": 1}],
        consensus={"diagnosis": "高血压"},
        divergences=[],
    )
    assert result.final_report == "# 报告"
    assert result.status == "completed"


def test_consultation_result_pending():
    result = ConsultationResult(
        id="c-456",
        status="running",
        final_report=None,
        discussion_rounds=[],
        consensus=None,
        divergences=[],
    )
    assert result.final_report is None


def test_consultation_status_enum():
    assert ConsultationStatus.PENDING == "pending"
    assert ConsultationStatus.RUNNING == "running"
    assert ConsultationStatus.COMPLETED == "completed"
    assert ConsultationStatus.FAILED == "failed"


def test_knowledge_search_request():
    req = KnowledgeSearchRequest(query="高血压治疗方案")
    assert req.top_k == 10


def test_search_response():
    resp = SearchResponse(
        results=[{"text": "高血压...", "score": 0.9}],
        total=1,
    )
    assert resp.total == 1


def test_ingest_response():
    resp = IngestResponse(job_id="j-001", status="processing", message="已提交")
    assert resp.job_id == "j-001"


def test_consultation_store_create():
    store = ConsultationStore()
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    cid = store.create(req)
    assert cid.startswith("mdt-")
    entry = store.get(cid)
    assert entry is not None
    assert entry["status"] == "pending"


def test_consultation_store_update():
    store = ConsultationStore()
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    cid = store.create(req)
    store.update(cid, status="running")
    entry = store.get(cid)
    assert entry["status"] == "running"


def test_consultation_store_get_nonexistent():
    store = ConsultationStore()
    assert store.get("nonexistent") is None


def test_consultation_store_list():
    store = ConsultationStore()
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    store.create(req)
    store.create(req)
    all_items = store.list_all()
    assert len(all_items) == 2
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement models.py and deps.py**

```python
# src/medmdt/api/models.py
import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class ConsultationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsultationRequest(BaseModel):
    patient_info: dict
    medical_records: list[dict] = Field(default_factory=list)
    max_rounds: int = 3


class ConsultationResponse(BaseModel):
    id: str
    status: str
    created_at: str


class ConsultationResult(BaseModel):
    id: str
    status: str
    final_report: str | None = None
    discussion_rounds: list[dict] = Field(default_factory=list)
    consensus: dict | None = None
    divergences: list[str] = Field(default_factory=list)


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    results: list[dict]
    total: int


class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ConsultationStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def create(self, request: ConsultationRequest) -> str:
        cid = f"mdt-{uuid.uuid4().hex[:12]}"
        self._store[cid] = {
            "id": cid,
            "status": ConsultationStatus.PENDING,
            "patient_info": request.patient_info,
            "medical_records": request.medical_records,
            "max_rounds": request.max_rounds,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_report": None,
            "discussion_rounds": [],
            "consensus": None,
            "divergences": [],
        }
        return cid

    def get(self, consultation_id: str) -> dict | None:
        return self._store.get(consultation_id)

    def update(self, consultation_id: str, **fields) -> None:
        if consultation_id in self._store:
            self._store[consultation_id].update(fields)

    def list_all(self) -> list[dict]:
        return list(self._store.values())
```

```python
# src/medmdt/api/deps.py
from functools import lru_cache
from medmdt.api.models import ConsultationStore


@lru_cache
def get_consultation_store() -> ConsultationStore:
    return ConsultationStore()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_models.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/medmdt/api/ tests/test_api_models.py pyproject.toml
git commit -m "feat: API models and consultation store — request/response schemas for gateway"
```

---

### Task 2: Consultation Routes

**Files:**
- Create: `src/medmdt/api/routes/__init__.py`
- Create: `src/medmdt/api/routes/consultation.py`
- Test: `tests/test_api_consultation.py`

**Interfaces:**
- Consumes: `ConsultationRequest`, `ConsultationResponse`, `ConsultationResult`, `ConsultationStore`, `ConsultationStatus` from Task 1; `build_mdt_graph`, `run_consultation` from P2; `create_all_experts`, `Moderator`, `FusionRetriever`, `create_chat_model`, `get_settings` from P1/P2
- Produces:
  - `router = APIRouter(prefix="/api/v1/consultation", tags=["consultation"])`
  - `POST /` — creates consultation, starts background task, returns ConsultationResponse
  - `GET /{consultation_id}` — returns ConsultationResult
  - `GET /` — lists all consultations

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_consultation.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medmdt.api.routes.consultation import router, run_consultation_task
from medmdt.api.models import ConsultationStore
from medmdt.api.deps import get_consultation_store


@pytest.fixture
def store():
    return ConsultationStore()


@pytest.fixture
def app(store):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_consultation_store] = lambda: store
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_create_consultation(client):
    response = client.post("/api/v1/consultation", json={
        "patient_info": {"chief_complaint": "头痛"},
        "medical_records": [{"type": "vital", "content": "BP 160/100"}],
    })
    assert response.status_code == 201
    data = response.json()
    assert data["id"].startswith("mdt-")
    assert data["status"] == "pending"
    assert "created_at" in data


def test_create_consultation_minimal(client):
    response = client.post("/api/v1/consultation", json={
        "patient_info": {"chief_complaint": "发热"},
    })
    assert response.status_code == 201


def test_get_consultation_not_found(client):
    response = client.get("/api/v1/consultation/nonexistent")
    assert response.status_code == 404


def test_get_consultation_exists(client, store):
    from medmdt.api.models import ConsultationRequest
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    cid = store.create(req)
    store.update(cid, status="completed", final_report="# 报告")

    response = client.get(f"/api/v1/consultation/{cid}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == cid
    assert data["status"] == "completed"
    assert data["final_report"] == "# 报告"


def test_list_consultations(client, store):
    from medmdt.api.models import ConsultationRequest
    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    store.create(req)
    store.create(req)

    response = client.get("/api/v1/consultation")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_list_consultations_empty(client):
    response = client.get("/api/v1/consultation")
    assert response.status_code == 200
    assert response.json() == []
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_consultation.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement consultation routes**

```python
# src/medmdt/api/routes/consultation.py
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from medmdt.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    ConsultationResult,
    ConsultationStatus,
    ConsultationStore,
)
from medmdt.api.deps import get_consultation_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consultation", tags=["consultation"])


def run_consultation_task(consultation_id: str, store: ConsultationStore) -> None:
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.mdt.experts.factory import create_all_experts
    from medmdt.mdt.moderator import Moderator
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever
    from medmdt.mdt.graph import build_mdt_graph, run_consultation

    entry = store.get(consultation_id)
    if not entry:
        return

    store.update(consultation_id, status=ConsultationStatus.RUNNING)

    try:
        settings = get_settings()
        llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)

        experts = create_all_experts("config/experts.yaml")
        moderator = Moderator(llm=llm, consensus_threshold=settings.mdt_consensus_threshold)

        graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        vector_store = VectorStore(
            settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
        )
        keyword_store = KeywordStore(settings.elasticsearch_url)

        def embed_fn(texts):
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=settings.embedding_model)
            return embeddings.embed_documents(texts)

        retriever = FusionRetriever(graph_store, vector_store, keyword_store, llm, embed_fn)

        graph = build_mdt_graph(experts, moderator, retriever, llm)
        result = run_consultation(
            graph,
            patient_info=entry["patient_info"],
            medical_records=entry["medical_records"],
            max_rounds=entry["max_rounds"],
        )

        store.update(
            consultation_id,
            status=ConsultationStatus.COMPLETED,
            final_report=result["final_report"],
            discussion_rounds=result["discussion_rounds"],
            consensus=result["consensus"],
            divergences=result["divergences"],
        )
    except Exception as e:
        logger.exception("Consultation %s failed", consultation_id)
        store.update(consultation_id, status=ConsultationStatus.FAILED)


@router.post("", status_code=201, response_model=ConsultationResponse)
def create_consultation(
    request: ConsultationRequest,
    background_tasks: BackgroundTasks,
    store: ConsultationStore = Depends(get_consultation_store),
):
    cid = store.create(request)
    entry = store.get(cid)
    background_tasks.add_task(run_consultation_task, cid, store)
    return ConsultationResponse(
        id=cid, status=entry["status"], created_at=entry["created_at"],
    )


@router.get("", response_model=list[ConsultationResult])
def list_consultations(
    store: ConsultationStore = Depends(get_consultation_store),
):
    items = store.list_all()
    return [
        ConsultationResult(
            id=item["id"],
            status=item["status"],
            final_report=item.get("final_report"),
            discussion_rounds=item.get("discussion_rounds", []),
            consensus=item.get("consensus"),
            divergences=item.get("divergences", []),
        )
        for item in items
    ]


@router.get("/{consultation_id}", response_model=ConsultationResult)
def get_consultation(
    consultation_id: str,
    store: ConsultationStore = Depends(get_consultation_store),
):
    entry = store.get(consultation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return ConsultationResult(
        id=entry["id"],
        status=entry["status"],
        final_report=entry.get("final_report"),
        discussion_rounds=entry.get("discussion_rounds", []),
        consensus=entry.get("consensus"),
        divergences=entry.get("divergences", []),
    )
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_consultation.py -v`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/api/routes/ tests/test_api_consultation.py
git commit -m "feat: consultation routes — create, get, list MDT consultations with background execution"
```

---

### Task 3: Knowledge Routes

**Files:**
- Create: `src/medmdt/api/routes/knowledge.py`
- Test: `tests/test_api_knowledge.py`

**Interfaces:**
- Consumes: `KnowledgeSearchRequest`, `SearchResponse`, `IngestResponse` from Task 1; `FusionRetriever.retrieve()` from P1; `ExtractionAgent.process_file()` from P1
- Produces:
  - `router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])`
  - `POST /search` — searches the knowledge base, returns SearchResponse
  - `POST /ingest` — accepts file upload, starts background ingestion, returns IngestResponse
  - `GET /ingest/{job_id}` — queries ingestion job status

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_knowledge.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medmdt.api.routes.knowledge import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@patch("medmdt.api.routes.knowledge._get_retriever")
def test_search(mock_get_retriever, client):
    mock_retriever = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "高血压诊断标准..."
    mock_result.score = 0.92
    mock_result.source = "vector"
    mock_result.metadata = {"section": "诊断"}
    mock_retriever.retrieve.return_value = [mock_result]
    mock_get_retriever.return_value = mock_retriever

    response = client.post("/api/v1/knowledge/search", json={
        "query": "高血压诊断标准",
        "top_k": 5,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["text"] == "高血压诊断标准..."
    mock_retriever.retrieve.assert_called_once_with("高血压诊断标准", top_k=5)


@patch("medmdt.api.routes.knowledge._get_retriever")
def test_search_empty(mock_get_retriever, client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_get_retriever.return_value = mock_retriever

    response = client.post("/api/v1/knowledge/search", json={
        "query": "nonexistent",
    })
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_ingest_no_file(client):
    response = client.post("/api/v1/knowledge/ingest")
    assert response.status_code == 422


def test_get_ingest_status_not_found(client):
    response = client.get("/api/v1/knowledge/ingest/nonexistent")
    assert response.status_code == 404
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_knowledge.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement knowledge routes**

```python
# src/medmdt/api/routes/knowledge.py
import logging
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from medmdt.api.models import KnowledgeSearchRequest, SearchResponse, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

_ingest_jobs: dict[str, dict] = {}


def _get_retriever():
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever

    settings = get_settings()
    llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)
    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore(
        settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
    )
    keyword_store = KeywordStore(settings.elasticsearch_url)

    def embed_fn(texts):
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        return embeddings.embed_documents(texts)

    return FusionRetriever(graph_store, vector_store, keyword_store, llm, embed_fn)


def _run_ingest(job_id: str, file_path: str) -> None:
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.extractor.agent import ExtractionAgent

    try:
        _ingest_jobs[job_id]["status"] = "processing"
        settings = get_settings()
        llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)

        graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        vector_store = VectorStore(
            settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
        )
        vector_store.ensure_collection()
        keyword_store = KeywordStore(settings.elasticsearch_url)
        keyword_store.ensure_index()

        def embed_fn(texts):
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=settings.embedding_model)
            return embeddings.embed_documents(texts)

        agent = ExtractionAgent(
            settings=settings,
            graph_store=graph_store,
            vector_store=vector_store,
            keyword_store=keyword_store,
            embed_fn=embed_fn,
            llm=llm,
        )
        reports = agent.process_file(file_path)
        total_entities = sum(r.entities_count for r in reports)
        total_chunks = sum(r.chunks_count for r in reports)
        _ingest_jobs[job_id].update(
            status="completed",
            message=f"提取完成: {total_entities} 实体, {total_chunks} 文本块",
        )
    except Exception as e:
        logger.exception("Ingest job %s failed", job_id)
        _ingest_jobs[job_id].update(status="failed", message=str(e))
    finally:
        Path(file_path).unlink(missing_ok=True)


@router.post("/search", response_model=SearchResponse)
def search_knowledge(request: KnowledgeSearchRequest):
    retriever = _get_retriever()
    results = retriever.retrieve(request.query, top_k=request.top_k)
    return SearchResponse(
        results=[
            {
                "text": r.text,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata if hasattr(r, "metadata") else {},
            }
            for r in results
        ],
        total=len(results),
    )


@router.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id = f"ingest-{uuid.uuid4().hex[:12]}"
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": f"文件 {file.filename} 已接收",
        "filename": file.filename,
    }

    background_tasks.add_task(_run_ingest, job_id, tmp.name)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"文件 {file.filename} 已接收，开始处理",
    )


@router.get("/ingest/{job_id}", response_model=IngestResponse)
def get_ingest_status(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return IngestResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job.get("message", ""),
    )
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_knowledge.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/api/routes/knowledge.py tests/test_api_knowledge.py
git commit -m "feat: knowledge routes — search and ingest endpoints with background processing"
```

---

### Task 4: WebSocket Streaming for Consultation

**Files:**
- Create: `src/medmdt/api/routes/ws.py`
- Test: `tests/test_api_ws.py`

**Interfaces:**
- Consumes: `ConsultationStore` from Task 1; `get_consultation_store` from Task 1
- Produces:
  - `EventBus` — per-consultation event publisher/subscriber: `publish(consultation_id, event)`, `subscribe(consultation_id) -> asyncio.Queue`
  - `get_event_bus() -> EventBus` — singleton
  - `WebSocket /api/v1/consultation/{consultation_id}/stream` — streams JSON events to client

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_ws.py
import asyncio
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medmdt.api.routes.ws import router, EventBus, get_event_bus
from medmdt.api.models import ConsultationStore, ConsultationRequest
from medmdt.api.deps import get_consultation_store


@pytest.fixture
def store():
    s = ConsultationStore()
    req = ConsultationRequest(patient_info={"chief_complaint": "test"}, medical_records=[])
    s.create(req)
    return s


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def app(store, event_bus):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_consultation_store] = lambda: store
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_event_bus_publish_subscribe(event_bus):
    loop = asyncio.new_event_loop()

    async def run():
        queue = event_bus.subscribe("c-1")
        event_bus.publish("c-1", {"type": "test", "data": "hello"})
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg["type"] == "test"
        assert msg["data"] == "hello"

    loop.run_until_complete(run())
    loop.close()


def test_event_bus_multiple_subscribers(event_bus):
    loop = asyncio.new_event_loop()

    async def run():
        q1 = event_bus.subscribe("c-1")
        q2 = event_bus.subscribe("c-1")
        event_bus.publish("c-1", {"type": "test"})
        msg1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        msg2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert msg1["type"] == "test"
        assert msg2["type"] == "test"

    loop.run_until_complete(run())
    loop.close()


def test_event_bus_different_consultations(event_bus):
    loop = asyncio.new_event_loop()

    async def run():
        q1 = event_bus.subscribe("c-1")
        q2 = event_bus.subscribe("c-2")
        event_bus.publish("c-1", {"type": "for-c1"})
        msg1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        assert msg1["type"] == "for-c1"
        assert q2.empty()

    loop.run_until_complete(run())
    loop.close()


def test_websocket_consultation_not_found(client):
    with client.websocket_connect("/api/v1/consultation/nonexistent/stream") as ws:
        data = ws.receive_json()
        assert data["type"] == "error"


def test_websocket_receives_events(client, store, event_bus):
    cid = list(store._store.keys())[0]
    with client.websocket_connect(f"/api/v1/consultation/{cid}/stream") as ws:
        event_bus.publish(cid, {"type": "round_start", "round": 1})
        data = ws.receive_json()
        assert data["type"] == "round_start"
        assert data["round"] == 1

        event_bus.publish(cid, {"type": "done"})
        data = ws.receive_json()
        assert data["type"] == "done"
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_ws.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement WebSocket route and EventBus**

```python
# src/medmdt/api/routes/ws.py
import asyncio
import logging
from collections import defaultdict
from functools import lru_cache

from fastapi import APIRouter, WebSocket, Depends

from medmdt.api.models import ConsultationStore
from medmdt.api.deps import get_consultation_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, consultation_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[consultation_id].append(queue)
        return queue

    def unsubscribe(self, consultation_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(consultation_id, [])
        if queue in subs:
            subs.remove(queue)

    def publish(self, consultation_id: str, event: dict) -> None:
        for queue in self._subscribers.get(consultation_id, []):
            queue.put_nowait(event)


@lru_cache
def get_event_bus() -> EventBus:
    return EventBus()


@router.websocket("/api/v1/consultation/{consultation_id}/stream")
async def consultation_stream(
    websocket: WebSocket,
    consultation_id: str,
    store: ConsultationStore = Depends(get_consultation_store),
    bus: EventBus = Depends(get_event_bus),
):
    await websocket.accept()

    entry = store.get(consultation_id)
    if not entry:
        await websocket.send_json({"type": "error", "message": "Consultation not found"})
        await websocket.close()
        return

    queue = bus.subscribe(consultation_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event.get("type") == "done":
                break
    except Exception:
        logger.debug("WebSocket disconnected for %s", consultation_id)
    finally:
        bus.unsubscribe(consultation_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_ws.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/api/routes/ws.py tests/test_api_ws.py
git commit -m "feat: WebSocket streaming — real-time consultation event bus and endpoint"
```

---

### Task 5: FastAPI Application Assembly + Entry Point

**Files:**
- Create: `src/medmdt/api/app.py`
- Create: `scripts/server.py`
- Test: `tests/test_api_app.py`

**Interfaces:**
- Consumes: `router` from Task 2 (consultation), Task 3 (knowledge), Task 4 (ws)
- Produces:
  - `create_app() -> FastAPI` — assembles all routers into a FastAPI app
  - `scripts/server.py` — CLI entry point: `uvicorn medmdt.api.app:app`
  - Health check: `GET /health` returns `{"status": "ok"}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_app.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from medmdt.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_consultation_routes_mounted(client):
    response = client.post("/api/v1/consultation", json={
        "patient_info": {"chief_complaint": "头痛"},
    })
    assert response.status_code == 201


def test_knowledge_search_route_mounted(client):
    with patch("medmdt.api.routes.knowledge._get_retriever") as mock:
        mock_retriever = mock.return_value
        mock_retriever.retrieve.return_value = []
        response = client.post("/api/v1/knowledge/search", json={
            "query": "test", "top_k": 5,
        })
        assert response.status_code == 200


def test_openapi_docs(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    paths = list(spec["paths"].keys())
    assert "/health" in paths
    assert "/api/v1/consultation" in paths
    assert "/api/v1/knowledge/search" in paths
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_app.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement app.py**

```python
# src/medmdt/api/app.py
from fastapi import FastAPI

from medmdt.api.routes.consultation import router as consultation_router
from medmdt.api.routes.knowledge import router as knowledge_router
from medmdt.api.routes.ws import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedMDT",
        description="多专家会诊医学Agent系统API",
        version="0.1.0",
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    app.include_router(consultation_router)
    app.include_router(knowledge_router)
    app.include_router(ws_router)

    return app


app = create_app()
```

- [ ] **Step 3: Create server entry point**

```python
# scripts/server.py
"""Start the MedMDT API server.

Usage:
    python scripts/server.py [--host HOST] [--port PORT]
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="MedMDT API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    uvicorn.run(
        "medmdt.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_api_app.py -v`
Expected: 4 passed

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/medmdt/api/app.py scripts/server.py tests/test_api_app.py
git commit -m "feat: FastAPI application assembly — health check, all routers, server entry point"
```
