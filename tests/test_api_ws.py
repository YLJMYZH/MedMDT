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
