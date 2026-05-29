from dataclasses import dataclass
import re

from app.schemas.plans import Plan, PlanStep, PlanStepStatus

FALLBACK_MESSAGE = "I cannot do that yet. Available Stage 1 actions are: open Notepad, Calculator, Paint, Explorer."


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
                PlanStep(number=2, title="Select tool", status=PlanStepStatus.pending),
                PlanStep(number=3, title="Execute", status=PlanStepStatus.pending),
            ],
        )

    def propose_tool_call(self, user_message: str) -> ToolProposal | None:
        normalized = _normalize(user_message)
        if not normalized:
            return None

        if _is_tool_listing_request(normalized):
            return ToolProposal(name="list_available_tools", arguments={})

        if _is_time_request(normalized):
            return ToolProposal(name="get_current_time", arguments={})

        app_name = _match_app_request(normalized)
        if app_name is not None:
            return ToolProposal(name="launch_app", arguments={"app_name": app_name})

        return None

    @staticmethod
    def _goal_from_message(user_message: str) -> str:
        normalized = _normalize(user_message)
        if _is_tool_listing_request(normalized):
            return "List available tools"
        if _is_time_request(normalized):
            return "Get current time"
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


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_tool_listing_request(normalized: str) -> bool:
    return normalized in {
        "what tools do you have",
        "what tools are available",
        "list tools",
        "list available tools",
        "show tools",
        "show available tools",
        "what can you do",
    }


def _is_time_request(normalized: str) -> bool:
    return normalized in {
        "what time is it",
        "what is the current time",
        "current time",
        "get current time",
        "tell me the time",
        "time",
    }


def _match_app_request(normalized: str) -> str | None:
    app_aliases = {
        "notepad": "notepad",
        "calculator": "calculator",
        "calc": "calculator",
        "paint": "paint",
        "mspaint": "paint",
        "explorer": "explorer",
        "file explorer": "explorer",
        "windows explorer": "explorer",
    }

    for alias, app_name in app_aliases.items():
        if normalized == alias:
            return app_name
        pattern = rf"\b(open|launch|start|run)\s+(the\s+)?{re.escape(alias)}\b"
        if re.search(pattern, normalized):
            return app_name
        if _has_launch_intent(normalized) and re.search(rf"\b{re.escape(alias)}\b", normalized):
            return app_name

    return None


def _has_launch_intent(normalized: str) -> bool:
    return re.search(r"\b(open|launch|start|run)\b", normalized) is not None
