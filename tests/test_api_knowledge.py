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
