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
