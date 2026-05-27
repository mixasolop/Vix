import asyncio
import logging

from fastapi import WebSocket

from app.schemas.events import AssistantEvent

LOGGER = logging.getLogger("app.events")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._subscribers.add(websocket)
            subscriber_count = len(self._subscribers)
        LOGGER.info("websocket subscribed | subscribers=%s", subscriber_count)

    async def unsubscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._subscribers.discard(websocket)
            subscriber_count = len(self._subscribers)
        LOGGER.info("websocket unsubscribed | subscribers=%s", subscriber_count)

    async def publish(self, event: AssistantEvent) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        stale_subscribers: list[WebSocket] = []
        payload = event.model_dump(mode="json")
        LOGGER.info("websocket publish | type=%s | subscribers=%s", event.type, len(subscribers))
        for websocket in subscribers:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                stale_subscribers.append(websocket)

        if stale_subscribers:
            async with self._lock:
                for websocket in stale_subscribers:
                    self._subscribers.discard(websocket)
                subscriber_count = len(self._subscribers)
            LOGGER.info("removed stale websocket subscribers | subscribers=%s", subscriber_count)
