from __future__ import annotations

import json

from fastapi import APIRouter, Request

from app.schemas.browser import BrowserStatusResponse

router = APIRouter(prefix="/browser", tags=["browser"])


@router.get("/status", response_model=BrowserStatusResponse)
async def browser_status(request: Request) -> BrowserStatusResponse:
    manager = request.app.state.browser_manager
    snapshot = manager.get_active_snapshot_if_any()
    artifact = _latest_browser_artifact(request.app.state.database.list_artifacts(limit=50))
    return BrowserStatusResponse(
        current_url=snapshot.url if snapshot else None,
        page_title=snapshot.title if snapshot else None,
        text_preview=snapshot.text_preview if snapshot else None,
        links=snapshot.links if snapshot else [],
        forms=snapshot.forms if snapshot else [],
        last_browser_artifact=artifact,
        last_browser_action=_last_browser_action(artifact),
        risk_classification=_risk_classification(artifact),
        raw_snapshot=snapshot.model_dump(mode="json") if snapshot else None,
    )


def _latest_browser_artifact(records) -> dict[str, object] | None:
    for record in records:
        if not record.type.startswith("browser_") and record.type != "form_draft":
            continue
        return {
            "id": record.id,
            "type": record.type,
            "title": record.title,
            "content_text": record.content_text,
            "data": json.loads(record.data_json),
            "created_at": record.created_at.isoformat(),
        }
    return None


def _last_browser_action(artifact: dict[str, object] | None) -> str | None:
    if artifact is None:
        return None
    data = artifact.get("data")
    if isinstance(data, dict):
        return str(data.get("tool") or artifact.get("type") or "")
    return str(artifact.get("type") or "")


def _risk_classification(artifact: dict[str, object] | None) -> str | None:
    if artifact is None:
        return None
    data = artifact.get("data")
    if isinstance(data, dict):
        return str(data.get("risk_level") or data.get("risk_hint") or "unknown")
    return "unknown"
