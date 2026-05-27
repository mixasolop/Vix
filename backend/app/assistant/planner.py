from dataclasses import dataclass

from app.schemas.plans import Plan, PlanStep, PlanStepStatus


@dataclass(frozen=True)
class ToolProposal:
    name: str
    arguments: dict[str, object]


class Planner:
    def create_plan(self, user_message: str) -> Plan:
        goal = self._goal_from_message(user_message)
        return Plan(
            goal=goal,
            steps=[
                PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                PlanStep(number=2, title="Select tool", status=PlanStepStatus.completed),
                PlanStep(number=3, title="Execute", status=PlanStepStatus.completed),
            ],
        )

    def propose_tool_call(self, user_message: str) -> ToolProposal | None:
        normalized = user_message.strip().lower()
        if not normalized:
            return None

        app_aliases = {
            "notepad": "notepad",
            "calculator": "calculator",
            "calc": "calculator",
            "paint": "paint",
            "mspaint": "paint",
            "explorer": "explorer",
        }
        for alias, app_name in app_aliases.items():
            if alias in normalized:
                return ToolProposal(name="launch_app", arguments={"app_name": app_name})

        if normalized.startswith("open "):
            return ToolProposal(name="launch_app", arguments={"app_name": normalized.removeprefix("open ").strip()})

        return None

    @staticmethod
    def _goal_from_message(user_message: str) -> str:
        normalized = user_message.strip().lower()
        if "notepad" in normalized:
            return "Open Notepad"
        if "calculator" in normalized or "calc" in normalized:
            return "Open Calculator"
        if "paint" in normalized or "mspaint" in normalized:
            return "Open Paint"
        if "explorer" in normalized:
            return "Open Explorer"
        if normalized:
            return f"Handle request: {user_message.strip()}"
        return "Handle empty request"
