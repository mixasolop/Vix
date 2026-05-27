from fastapi import APIRouter, Request

from app.assistant.policy import PermissionManager
from app.schemas.tools import PermissionDecisionResponse

router = APIRouter(prefix="/permissions", tags=["permissions"])


@router.post("/{permission_id}/approve", response_model=PermissionDecisionResponse)
async def approve_permission(permission_id: str, request: Request) -> PermissionDecisionResponse:
    permission_manager: PermissionManager = request.app.state.permission_manager
    return permission_manager.approve(permission_id)


@router.post("/{permission_id}/reject", response_model=PermissionDecisionResponse)
async def reject_permission(permission_id: str, request: Request) -> PermissionDecisionResponse:
    permission_manager: PermissionManager = request.app.state.permission_manager
    return permission_manager.reject(permission_id)
