from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.events.event_bus import EventBus
from app.schemas.events import AssistantEvent

router = APIRouter(tags=["events"])


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    event_bus: EventBus = websocket.app.state.event_bus
    await event_bus.subscribe(websocket)

    ready_event = AssistantEvent(
        type="event_stream.connected",
        payload={"message": "WebSocket event stream ready"},
    )
    await websocket.send_json(ready_event.model_dump(mode="json"))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await event_bus.unsubscribe(websocket)
        return
