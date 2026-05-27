from app.schemas.tools import PermissionDecisionResponse


class PermissionManager:
    def __init__(self, database=None) -> None:
        self._decisions: dict[str, str] = {}
        self._database = database

    def approve(self, permission_id: str) -> PermissionDecisionResponse:
        self._decisions[permission_id] = "approved"
        if self._database is not None:
            self._database.update_permission_status(permission_id, "approved")
        return PermissionDecisionResponse(permission_id=permission_id, status="approved")

    def reject(self, permission_id: str) -> PermissionDecisionResponse:
        self._decisions[permission_id] = "rejected"
        if self._database is not None:
            self._database.update_permission_status(permission_id, "rejected")
        return PermissionDecisionResponse(permission_id=permission_id, status="rejected")

    def get_status(self, permission_id: str) -> str | None:
        return self._decisions.get(permission_id)
