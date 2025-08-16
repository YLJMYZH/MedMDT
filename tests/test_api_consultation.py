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


def test_run_consultation_task_publishes_events(store):
    """Verify run_consultation_task publishes EventBus events on success and failure."""
    from medmdt.api.models import ConsultationRequest, ConsultationStatus
    from medmdt.api.routes.ws import EventBus

    req = ConsultationRequest(
        patient_info={"chief_complaint": "test"}, medical_records=[],
    )
    cid = store.create(req)

    # Test success path
    bus = EventBus()
    queue = bus.subscribe(cid)

    with patch("medmdt.api.routes.ws.get_event_bus", return_value=bus), \
         patch("medmdt.api.deps.build_infrastructure") as mock_infra, \
         patch("medmdt.mdt.experts.factory.create_all_experts") as mock_experts, \
         patch("medmdt.mdt.moderator.Moderator") as mock_mod, \
         patch("medmdt.mdt.graph.build_mdt_graph") as mock_graph, \
         patch("medmdt.mdt.graph.run_consultation") as mock_run:

        mock_infra.return_value = {
            "settings": MagicMock(mdt_consensus_threshold=0.8),
            "llm": MagicMock(),
            "retriever": MagicMock(),
        }
        mock_run.return_value = {
            "final_report": "report",
            "discussion_rounds": [],
            "consensus": {},
            "divergences": [],
        }
        run_consultation_task(cid, store)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert len(events) == 2
    assert events[0] == {"type": "status", "status": "running"}
    assert events[1] == {"type": "done", "status": "completed"}

    # Test failure path
    cid2 = store.create(req)
    bus2 = EventBus()
    queue2 = bus2.subscribe(cid2)

    with patch("medmdt.api.routes.ws.get_event_bus", return_value=bus2), \
         patch("medmdt.api.deps.build_infrastructure", side_effect=RuntimeError("boom")):
        run_consultation_task(cid2, store)

    events2 = []
    while not queue2.empty():
        events2.append(queue2.get_nowait())

    assert len(events2) == 3
    assert events2[0] == {"type": "status", "status": "running"}
    assert events2[1] == {"type": "error", "message": "boom"}
    assert events2[2] == {"type": "done", "status": "failed"}
