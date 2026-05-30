from dataclasses import dataclass
import re
from typing import Literal

from app.assistant.text_canonicalization import canonicalize_user_text
from app.schemas.plans import Plan, PlanStep, PlanStepStatus

FALLBACK_MESSAGE = "I cannot do that yet. Available Stage 1 actions are: open Notepad, Calculator, Paint, Explorer."


@dataclass(frozen=True)
class ToolProposal:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ActionDecision:
    intent_type: Literal[
        "EXECUTE_TOOL",
        "ANSWER_ONLY",
        "NO_ACTION",
        "ASK_CLARIFICATION",
        "MISSING_CAPABILITY",
        "BLOCKED",
    ]
    reason: str
    confidence: float
    tool_proposal: ToolProposal | None = None
    missing_input: str | None = None


class Planner:
    def create_plan(self, user_message: str) -> Plan:
        goal = self._goal_from_message(user_message)
        return Plan(
            goal=goal,
            steps=[
                PlanStep(number=1, key="understand_action", title="Understand action request", status=PlanStepStatus.completed),
                PlanStep(number=2, key="select_tool", title="Select tool", status=PlanStepStatus.pending),
                PlanStep(number=3, key="execute_tool", title="Execute tool", status=PlanStepStatus.pending),
            ],
        )

    def propose_tool_call(self, user_message: str) -> ToolProposal | None:
        if len(_split_sequenced_actions(user_message)) > 1:
            return None
        decision = self.decide_action(user_message)
        return decision.tool_proposal if decision.intent_type == "EXECUTE_TOOL" else None

    def propose_tool_sequence(self, user_message: str) -> list[ToolProposal]:
        segments = _split_sequenced_actions(user_message)
        if len(segments) < 2:
            return []

        proposals: list[ToolProposal] = []
        for segment in segments:
            decision = self.decide_action(segment)
            if decision.intent_type != "EXECUTE_TOOL" or decision.tool_proposal is None:
                return []
            proposals.append(decision.tool_proposal)
        return proposals

    def decide_action(self, user_message: str) -> ActionDecision:
        normalized = _normalize(user_message)
        if not normalized:
            return ActionDecision("ANSWER_ONLY", "Empty request should receive a normal assistant reply.", 0.9)

        if _is_tool_listing_request(normalized):
            return ActionDecision(
                "EXECUTE_TOOL",
                "User asked to list available tools.",
                0.98,
                ToolProposal(name="list_available_tools", arguments={}),
            )

        if _is_time_request(normalized):
            return ActionDecision(
                "EXECUTE_TOOL",
                "User asked for the current time.",
                0.98,
                ToolProposal(name="get_current_time", arguments={}),
            )

        app_name = _mentioned_app(normalized)
        if app_name is not None:
            label = _app_label(app_name)
            if _has_negated_action_intent(normalized):
                return ActionDecision("NO_ACTION", f"Okay, I will not open {label}.", 0.99)
            if _is_hypothetical_or_instructional_question(normalized):
                return ActionDecision("ANSWER_ONLY", f"User is asking about opening {label}, not asking to run it.", 0.95)
            if _is_ambiguous_app_reference(normalized):
                return ActionDecision(
                    "ASK_CLARIFICATION",
                    f"Do you want me to open {label}?",
                    0.85,
                    ToolProposal(name="launch_app", arguments={"app_name": app_name}),
                    missing_input="confirmation",
                )
            if _has_affirmative_launch_intent(normalized):
                return ActionDecision(
                    "EXECUTE_TOOL",
                    f"User asked to open {label}.",
                    0.98,
                    ToolProposal(name="launch_app", arguments={"app_name": app_name}),
                )

        if _has_launch_intent(normalized):
            return ActionDecision("MISSING_CAPABILITY", "User asked to launch an unsupported or unknown app.", 0.75)

        return ActionDecision("ANSWER_ONLY", "No deterministic local action intent detected.", 0.35)

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
    return canonicalize_user_text(value)


def _split_sequenced_actions(value: str) -> list[str]:
    normalized = _normalize(value)
    if not normalized:
        return []
    parts = re.split(r"\s+(?:and then|then|and)\s+", normalized)
    return [part.strip(" .,;") for part in parts if part.strip(" .,;")]


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
    decision = Planner().decide_action(normalized)
    if decision.intent_type == "EXECUTE_TOOL" and decision.tool_proposal is not None:
        app_name = decision.tool_proposal.arguments.get("app_name")
        return str(app_name) if app_name is not None else None
    return None


def _mentioned_app(normalized: str) -> str | None:
    app_aliases = {
        "windows explorer": "explorer",
        "file explorer": "explorer",
        "calculator": "calculator",
        "notepad": "notepad",
        "mspaint": "paint",
        "explorer": "explorer",
        "paint": "paint",
        "calc": "calculator",
    }

    for alias, app_name in app_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return app_name

    return None


def _has_launch_intent(normalized: str) -> bool:
    return re.search(r"\b(open|launch|start|run)\b", normalized) is not None


def _has_affirmative_launch_intent(normalized: str) -> bool:
    if _has_negated_action_intent(normalized):
        return False
    if _is_hypothetical_or_instructional_question(normalized):
        return False
    return _has_launch_intent(normalized)


def _has_negated_action_intent(normalized: str) -> bool:
    negation_patterns = (
        r"\b(don't|dont|do not|never|please don't|please dont|stop,? don't|stop,? dont)\s+(ever\s+)?(open|launch|start|run)\b",
        r"\bi\s+(don't|dont|do not)\s+want\s+(you\s+)?to\s+(open|launch|start|run)\b",
        r"\bdon't\s+want\s+(you\s+)?to\s+(open|launch|start|run)\b",
        r"\bdo\s+not\s+want\s+(you\s+)?to\s+(open|launch|start|run)\b",
    )
    return any(re.search(pattern, normalized) for pattern in negation_patterns)


def _is_hypothetical_or_instructional_question(normalized: str) -> bool:
    question_patterns = (
        r"^(what happens if|what would happen if|what if)\s+(i|you|we|someone)\s+(open|launch|start|run)\b",
        r"^(how do i|how can i|how would i|how should i)\s+(open|launch|start|run)\b",
        r"^(can i|could i|should i|would it)\b.*\b(open|launch|start|run)\b",
    )
    return any(re.search(pattern, normalized) for pattern in question_patterns)


def _is_ambiguous_app_reference(normalized: str) -> bool:
    normalized = normalized.strip(" ?.!.,")
    if normalized in {"notepad", "calculator", "calc", "paint", "mspaint", "explorer", "file explorer", "windows explorer"}:
        return True
    return re.match(r"^(maybe|perhaps|possibly)\s+(notepad|calculator|calc|paint|mspaint|explorer|file explorer|windows explorer)\??$", normalized) is not None


def _app_label(app_name: str) -> str:
    return {
        "calculator": "Calculator",
        "notepad": "Notepad",
        "paint": "Paint",
        "explorer": "Explorer",
    }.get(app_name, app_name)
