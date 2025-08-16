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
