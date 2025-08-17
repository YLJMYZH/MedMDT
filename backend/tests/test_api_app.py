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
