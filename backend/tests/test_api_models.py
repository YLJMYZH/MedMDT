import pytest
from pydantic import ValidationError
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


def test_max_rounds_lower_bound():
    """max_rounds must be >= 1."""
    with pytest.raises(ValidationError):
        ConsultationRequest(
            patient_info={"chief_complaint": "test"},
            medical_records=[],
            max_rounds=0,
        )


def test_max_rounds_upper_bound():
    """max_rounds must be <= 10."""
    with pytest.raises(ValidationError):
        ConsultationRequest(
            patient_info={"chief_complaint": "test"},
            medical_records=[],
            max_rounds=11,
        )


def test_max_rounds_boundary_valid():
    """max_rounds at boundaries 1 and 10 should be valid."""
    req_min = ConsultationRequest(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        max_rounds=1,
    )
    assert req_min.max_rounds == 1

    req_max = ConsultationRequest(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        max_rounds=10,
    )
    assert req_max.max_rounds == 10
