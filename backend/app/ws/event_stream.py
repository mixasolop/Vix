from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.events import AssistantEvent

router = APIRouter(tags=["events"])


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    ready_event = AssistantEvent(
        type="event_stream.connected",
        payload={"message": "WebSocket event stream ready"},
    )
    await websocket.send_json(ready_event.model_dump(mode="json"))

    try:
        while True:
            incoming = await websocket.receive_json()
            echo_event = AssistantEvent(
                type="client.event.received",
                payload={"received": incoming},
            )
            await websocket.send_json(echo_event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
