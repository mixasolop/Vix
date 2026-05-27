from dataclasses import dataclass

from app.schemas.tools import PermissionDecisionResponse


@dataclass(frozen=True)
class PendingAction:
    permission_id: str
    session_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, object]


class PermissionManager:
    def __init__(self, database=None) -> None:
        self._decisions: dict[str, str] = {}
        self._pending_actions: dict[str, PendingAction] = {}
        self._database = database

    def add_pending_action(self, action: PendingAction) -> None:
        self._pending_actions[action.permission_id] = action

    def approve(self, permission_id: str) -> tuple[PermissionDecisionResponse, PendingAction | None]:
        self._decisions[permission_id] = "approved"
        if self._database is not None:
            self._database.update_permission_status(permission_id, "approved")
        return PermissionDecisionResponse(permission_id=permission_id, status="approved"), self._pending_actions.pop(permission_id, None)

    def reject(self, permission_id: str) -> tuple[PermissionDecisionResponse, PendingAction | None]:
        self._decisions[permission_id] = "rejected"
        if self._database is not None:
            self._database.update_permission_status(permission_id, "rejected")
        return PermissionDecisionResponse(permission_id=permission_id, status="rejected"), self._pending_actions.pop(permission_id, None)

    def get_status(self, permission_id: str) -> str | None:
        return self._decisions.get(permission_id)
