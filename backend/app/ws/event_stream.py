import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.events.event_bus import EventBus
from app.schemas.events import AssistantEvent

router = APIRouter(tags=["events"])
LOGGER = logging.getLogger("app.ws")


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    LOGGER.info("websocket accepted")
    event_bus: EventBus = websocket.app.state.event_bus
    await event_bus.subscribe(websocket)

    ready_event = AssistantEvent(
        type="event_stream_connected",
        data={"message": "WebSocket event stream ready"},
    )
    await websocket.send_json(ready_event.model_dump(mode="json"))

    try:
        while True:
            client_message = await websocket.receive_text()
            LOGGER.info("websocket client message ignored | length=%s", len(client_message))
    except WebSocketDisconnect:
        await event_bus.unsubscribe(websocket)
        LOGGER.info("websocket disconnected")
        return
