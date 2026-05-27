import asyncio

from fastapi import WebSocket

from app.schemas.events import AssistantEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._subscribers.add(websocket)

    async def unsubscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._subscribers.discard(websocket)

    async def publish(self, event: AssistantEvent) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        stale_subscribers: list[WebSocket] = []
        payload = event.model_dump(mode="json")
        for websocket in subscribers:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                stale_subscribers.append(websocket)

        if stale_subscribers:
            async with self._lock:
                for websocket in stale_subscribers:
                    self._subscribers.discard(websocket)
