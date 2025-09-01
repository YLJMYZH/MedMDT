# tests/test_api_app.py
import asyncio
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


def test_app_lifespan_closes_shared_llm_http_clients():
    from medmdt.llm.http_clients import (
        close_shared_http_clients,
        get_shared_async_http_client,
        get_shared_http_client,
    )

    asyncio.run(close_shared_http_clients())
    sync_client = get_shared_http_client()
    async_client = get_shared_async_http_client()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200

        assert sync_client.is_closed
        assert async_client.is_closed
    finally:
        asyncio.run(close_shared_http_clients())
