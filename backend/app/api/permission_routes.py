from fastapi import APIRouter, Request

from app.assistant.orchestrator import Orchestrator
from app.schemas.tools import PermissionDecisionResponse

router = APIRouter(prefix="/permissions", tags=["permissions"])


@router.post("/{permission_id}/approve", response_model=PermissionDecisionResponse)
async def approve_permission(permission_id: str, request: Request) -> PermissionDecisionResponse:
    orchestrator: Orchestrator = request.app.state.orchestrator
    return await orchestrator.approve_permission(permission_id)


@router.post("/{permission_id}/reject", response_model=PermissionDecisionResponse)
async def reject_permission(permission_id: str, request: Request) -> PermissionDecisionResponse:
    orchestrator: Orchestrator = request.app.state.orchestrator
    return await orchestrator.reject_permission(permission_id)
