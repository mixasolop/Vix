from dataclasses import dataclass
from enum import StrEnum
import re

from app.assistant.planner import Planner, ToolProposal


class RequestCategory(StrEnum):
    general_answer = "GENERAL_ANSWER"
    local_tool = "LOCAL_TOOL"
    realtime_info = "REALTIME_INFO"
    missing_tool = "MISSING_TOOL"
    unsafe_or_blocked = "UNSAFE_OR_BLOCKED"


@dataclass(frozen=True)
class RequestClassification:
    category: RequestCategory
    reason: str
    tool_proposal: ToolProposal | None = None
    missing_input: str | None = None


def classify_request(user_message: str, planner: Planner | None = None) -> RequestClassification:
    normalized = _normalize(user_message)
    planner = planner or Planner()

    if not normalized:
        return RequestClassification(RequestCategory.general_answer, "Empty request should receive a normal assistant reply.")

    if _looks_unsafe(normalized):
        return RequestClassification(RequestCategory.unsafe_or_blocked, "Request appears to ask for unsafe or destructive behavior.")

    weather_proposal, missing_input = _weather_tool_proposal(user_message)
    if weather_proposal is not None or missing_input is not None:
        return RequestClassification(
            RequestCategory.realtime_info,
            "Weather is real-time information and must use the weather tool.",
            weather_proposal,
            missing_input,
        )

    local_tool_proposal = planner.propose_tool_call(user_message)
    if local_tool_proposal is not None:
        return RequestClassification(
            RequestCategory.local_tool,
            "Request maps to an implemented deterministic local tool.",
            local_tool_proposal,
        )

    if _looks_like_missing_capability(normalized):
        return RequestClassification(
            RequestCategory.missing_tool,
            "Request requires a capability that is not implemented yet.",
        )

    return RequestClassification(RequestCategory.general_answer, "Request should be answered directly by the LLM.")


def _weather_tool_proposal(user_message: str) -> tuple[ToolProposal | None, str | None]:
    normalized = _normalize(user_message)
    if not _looks_like_weather_request(normalized):
        return None, None

    date = _extract_weather_date(normalized)
    location = _extract_weather_location(user_message)
    if location is None:
        return None, "location"
    return ToolProposal(name="get_weather", arguments={"location": location, "date": date}), None


def _extract_weather_date(normalized: str) -> str:
    if "tomorrow" in normalized:
        return "tomorrow"

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    if date_match:
        return date_match.group(0)

    return "today"


def _extract_weather_location(user_message: str) -> str | None:
    # Handles "weather today in Warsaw" and "weather in Warsaw today".
    match = re.search(
        r"\b(?:in|for|at)\s+([a-zA-Z][a-zA-Z\s.'-]*?)(?:\s+(?:today|tomorrow|on\s+\d{4}-\d{2}-\d{2}))?\??$",
        user_message.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    location = re.sub(r"\b(today|tomorrow)\b", "", match.group(1), flags=re.IGNORECASE).strip(" ?.,")
    return location or None


def _looks_like_weather_request(normalized: str) -> bool:
    return any(keyword in normalized for keyword in ("weather", "forecast", "temperature", "rain today", "raining today"))


def _looks_like_missing_capability(normalized: str) -> bool:
    missing_keywords = (
        "screen",
        "screenshot",
        "selected text",
        "clipboard",
        "look at",
        "read this",
        "find my",
        "find file",
        "search file",
        "search my files",
        "open file",
        "send message",
        "message anna",
        "summarize my",
        "email",
        "book",
        "order",
        "pay",
        "reminder",
        "calendar",
        "folder",
    )
    return any(keyword in normalized for keyword in missing_keywords) or _looks_like_unknown_local_action(normalized)


def should_propose_missing_tool_after_answer(user_message: str, assistant_message: str) -> bool:
    normalized_request = _normalize(user_message)
    normalized_answer = _normalize(assistant_message)
    return _looks_like_unsupported_action_request(normalized_request) and _looks_like_refusal_or_limitation(normalized_answer)


def _looks_like_unknown_local_action(normalized: str) -> bool:
    return re.match(r"^(open|launch|start|run)\s+\S+", normalized) is not None


def _looks_like_unsupported_action_request(normalized: str) -> bool:
    if _looks_like_missing_capability(normalized):
        return True

    action_verbs = (
        "automate",
        "change",
        "control",
        "copy",
        "create",
        "delete",
        "download",
        "edit",
        "manage",
        "move",
        "rename",
        "schedule",
        "set",
        "upload",
    )
    target_terms = (
        "app",
        "application",
        "computer",
        "desktop",
        "device",
        "drive",
        "file",
        "folder",
        "mixer",
        "music",
        "pc",
        "program",
        "setting",
        "settings",
        "smart home",
        "window",
    )
    return any(verb in normalized for verb in action_verbs) and any(target in normalized for target in target_terms)


def _looks_like_refusal_or_limitation(normalized: str) -> bool:
    normalized = normalized.replace("\u2019", "'")
    limitation_phrases = (
        "i can't",
        "i cannot",
        "i am not able",
        "i'm not able",
        "i don't have access",
        "i do not have access",
        "i can't directly",
        "i cannot directly",
        "not currently able",
        "not supported",
        "not implemented",
        "would need permission",
        "need confirmation",
        "requires confirmation",
        "instead",
        "alternative",
        "you can",
    )
    return any(phrase in normalized for phrase in limitation_phrases)


def _looks_unsafe(normalized: str) -> bool:
    unsafe_keywords = (
        "delete system32",
        "format my drive",
        "steal",
        "exfiltrate",
        "bypass permission",
    )
    return any(keyword in normalized for keyword in unsafe_keywords)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())
