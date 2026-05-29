import json

from fastapi import APIRouter, Request

from app.context.window_tracker import WindowTracker
from app.db.database import Database
from app.schemas.window_context import ContextStatusResponse

router = APIRouter(tags=["context"])


@router.get("/context/status", response_model=ContextStatusResponse)
async def context_status(request: Request) -> ContextStatusResponse:
    tracker: WindowTracker = request.app.state.context_tracker
    database: Database = request.app.state.database
    snapshot = tracker.snapshot()
    artifacts = database.list_artifacts(limit=1)
    last_artifact = None
    if artifacts:
        artifact = artifacts[0]
        last_artifact = {
            "id": artifact.id,
            "type": artifact.type,
            "title": artifact.title,
            "content_text": artifact.content_text,
            "data": json.loads(artifact.data_json),
            "created_at": artifact.created_at.isoformat(),
        }
    return ContextStatusResponse(
        current_foreground_window=snapshot.current_foreground_window,
        last_context_window=snapshot.last_context_window,
        last_context_artifact=last_artifact,
    )
