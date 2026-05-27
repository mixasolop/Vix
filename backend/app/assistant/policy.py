from dataclasses import dataclass

from app.schemas.tools import ConfirmationPolicy, PermissionDecisionResponse, RiskLevel, ToolDefinition, ToolStatus


@dataclass(frozen=True)
class PendingAction:
    permission_id: str
    session_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_permission: bool
    blocked: bool
    reason: str


class PolicyEngine:
    def evaluate_tool_call(self, tool: ToolDefinition, arguments: dict[str, object]) -> PolicyDecision:
        if tool.status != ToolStatus.implemented:
            return PolicyDecision(
                allowed=False,
                requires_permission=False,
                blocked=True,
                reason=f"Tool is not implemented: {tool.name}.",
            )

        if tool.risk_level in {RiskLevel.read, RiskLevel.low_write} and tool.confirmation_policy == ConfirmationPolicy.none:
            return PolicyDecision(
                allowed=True,
                requires_permission=False,
                blocked=False,
                reason=f"{tool.name} is allowed by Stage 1 policy.",
            )

        if tool.risk_level == RiskLevel.medium_write:
            return PolicyDecision(
                allowed=False,
                requires_permission=True,
                blocked=False,
                reason=f"{tool.name} is MEDIUM_WRITE and requires confirmation before execution.",
            )

        if tool.risk_level == RiskLevel.high_risk:
            return PolicyDecision(
                allowed=False,
                requires_permission=True,
                blocked=False,
                reason=f"{tool.name} is HIGH_RISK and requires confirmation before execution.",
            )

        if tool.confirmation_policy == ConfirmationPolicy.before_execute:
            return PolicyDecision(
                allowed=False,
                requires_permission=True,
                blocked=False,
                reason=f"{tool.name} requires confirmation before execution.",
            )

        return PolicyDecision(
            allowed=False,
            requires_permission=False,
            blocked=True,
            reason=f"{tool.name} has unsupported risk level: {tool.risk_level.value}.",
        )


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
