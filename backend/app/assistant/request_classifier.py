from dataclasses import dataclass, field
from enum import StrEnum
import re

from app.assistant.context_intent import ContextIntent, resolve_context_intent
from app.assistant.planner import ActionDecision, Planner, ToolProposal
from app.assistant.text_canonicalization import canonicalize_user_text


class RequestCategory(StrEnum):
    general_answer = "GENERAL_ANSWER"
    local_tool = "LOCAL_TOOL"
    realtime_info = "REALTIME_INFO"
    local_context = "LOCAL_CONTEXT"
    missing_tool = "MISSING_TOOL"
    multi_step = "MULTI_STEP"
    no_action = "NO_ACTION"
    ask_clarification = "ASK_CLARIFICATION"
    unsafe_or_blocked = "UNSAFE_OR_BLOCKED"


@dataclass(frozen=True)
class RequestClassification:
    category: RequestCategory
    reason: str
    tool_proposal: ToolProposal | None = None
    missing_input: str | None = None
    action_decision: ActionDecision | None = None
    context_intent: ContextIntent | None = None
    tool_sequence: list[ToolProposal] = field(default_factory=list)
    canonical_message: str = ""
    router_source: str = "deterministic"


def classify_request(user_message: str, planner: Planner | None = None) -> RequestClassification:
    normalized = canonicalize_user_text(user_message)
    planner = planner or Planner()

    # Classification order is part of the safety model. Higher-priority intents must
    # get the first chance to claim the request so a lower-priority keyword does not
    # accidentally trigger an action. For example, "don't open calculator" mentions a
    # valid local tool, but the explicit no-action intent must win before tool routing.
    #
    # Session-scoped pending clarification resolution happens in the orchestrator
    # immediately before this stateless classifier runs. That stateful resolver must
    # stay ahead of these keyword routes so "yes" only resolves an existing
    # clarification in the same session, and never becomes a free-standing action.

    # 1. Empty input: nothing to route, so let the assistant answer normally.
    if not normalized:
        return _classification(RequestCategory.general_answer, "Empty request should receive a normal assistant reply.", normalized)

    action_decision = planner.decide_action(user_message)

    # 2. Explicit no-action / negated action: this must beat unsafe, context, weather,
    # and local-tool checks because the user is explicitly telling us not to act.
    if action_decision.intent_type == "NO_ACTION":
        return RequestClassification(
            RequestCategory.no_action,
            action_decision.reason,
            action_decision=action_decision,
            canonical_message=normalized,
        )

    if _looks_like_negated_browser_action(normalized):
        return RequestClassification(
            RequestCategory.no_action,
            "Okay, no browser action taken.",
            canonical_message=normalized,
        )

    # 3. Unsafe/destructive requests: block before any context, realtime, or tool
    # handling can reinterpret a dangerous request as something benign.
    if _looks_unsafe(normalized):
        return _classification(
            RequestCategory.unsafe_or_blocked,
            "Request appears to ask for unsafe or destructive behavior.",
            normalized,
        )

    # 4. Local context requests: phrases like "selected word" can contain terms such as
    # "weather"; the context request must win so we inspect the selected text instead
    # of calling a realtime API for a word inside the selection.
    context_intent = resolve_context_intent(user_message)
    if context_intent is not None:
        return RequestClassification(
            RequestCategory.local_context,
            context_intent.reason,
            ToolProposal(context_intent.tool_name, context_intent.arguments),
            context_intent=context_intent,
            canonical_message=normalized,
        )

    browser_sequence = _browser_tool_sequence(user_message)
    if browser_sequence:
        return RequestClassification(
            RequestCategory.multi_step,
            "Request contains multiple browser actions and needs controlled step-by-step execution.",
            tool_sequence=browser_sequence,
            canonical_message=normalized,
        )

    browser_proposal, browser_missing_input, browser_reason = _browser_tool_proposal(user_message)
    if browser_proposal is not None or browser_missing_input is not None:
        return RequestClassification(
            RequestCategory.local_tool if browser_proposal is not None else RequestCategory.ask_clarification,
            browser_reason,
            browser_proposal,
            browser_missing_input,
            canonical_message=normalized,
        )

    # Mixed-intent guard: before choosing a single realtime or local tool, check whether
    # the deterministic task loop supports the full sequence. If not, ask clarification
    # rather than executing only one intent and silently dropping the other.
    tool_sequence = planner.propose_tool_sequence(user_message)
    if tool_sequence:
        return RequestClassification(
            RequestCategory.multi_step,
            "Request contains multiple deterministic tool actions and needs the task loop.",
            tool_sequence=tool_sequence,
            canonical_message=normalized,
        )

    mixed_intent = _mixed_intent_clarification(user_message, planner)
    if mixed_intent is not None:
        return mixed_intent

    # 5. Real-time information requests: weather/forecast/rain need tools instead of
    # model memory, but only after context and mixed-intent cases are ruled out.
    weather_proposal, missing_input = _weather_tool_proposal(user_message)
    if weather_proposal is not None or missing_input is not None:
        return RequestClassification(
            RequestCategory.realtime_info,
            "Weather is real-time information and must use the weather tool.",
            weather_proposal,
            missing_input,
            canonical_message=normalized,
        )

    # 6. Deterministic local tools: only affirmative execution decisions can become tool
    # calls. Hypothetical/instructional questions are kept as answer-only below.
    if action_decision.intent_type == "EXECUTE_TOOL" and action_decision.tool_proposal is not None:
        return RequestClassification(
            RequestCategory.local_tool,
            action_decision.reason,
            action_decision.tool_proposal,
            action_decision=action_decision,
            canonical_message=normalized,
        )

    if action_decision.intent_type == "ASK_CLARIFICATION":
        return RequestClassification(
            RequestCategory.ask_clarification,
            action_decision.reason,
            tool_proposal=action_decision.tool_proposal,
            missing_input=action_decision.missing_input,
            action_decision=action_decision,
            canonical_message=normalized,
        )

    if action_decision.intent_type == "BLOCKED":
        return RequestClassification(
            RequestCategory.unsafe_or_blocked,
            action_decision.reason,
            action_decision=action_decision,
            canonical_message=normalized,
        )

    # 7. Missing capability: requests for unsupported screen/file/message actions become
    # proposed-tool opportunities instead of vague chat answers.
    if action_decision.intent_type == "MISSING_CAPABILITY":
        return RequestClassification(
            RequestCategory.missing_tool,
            action_decision.reason,
            action_decision=action_decision,
            canonical_message=normalized,
        )

    if _looks_like_missing_capability(normalized):
        return RequestClassification(
            RequestCategory.missing_tool,
            "Request requires a capability that is not implemented yet.",
            canonical_message=normalized,
        )

    # 8. General answer: final safe fallback for explanations, how-to questions,
    # hypotheticals, and ordinary chat. No tool execution happens from this branch.
    return RequestClassification(
        RequestCategory.general_answer,
        action_decision.reason if action_decision.intent_type == "ANSWER_ONLY" else "Request should be answered directly by the LLM.",
        action_decision=action_decision,
        canonical_message=normalized,
    )


def _classification(category: RequestCategory, reason: str, canonical_message: str) -> RequestClassification:
    return RequestClassification(category, reason, canonical_message=canonical_message)


def _context_tool_proposal(normalized: str) -> ToolProposal | None:
    if _looks_like_selected_text_request(normalized):
        return ToolProposal(name="get_selected_text", arguments={})

    if any(phrase in normalized for phrase in ("foreground window", "technical foreground")):
        return ToolProposal(name="get_foreground_window_info", arguments={})

    if any(phrase in normalized for phrase in ("clipboard", "copied text", "copy buffer", "what did i copy")):
        return ToolProposal(name="get_clipboard_text", arguments={})

    if any(
        phrase in normalized
        for phrase in (
            "what app am i using",
            "what app was i using",
            "which app am i using",
            "which app was i using",
            "current window",
            "this window",
            "context window",
            "previous window",
        )
    ):
        return ToolProposal(name="get_context_window_info", arguments={})

    return None


def _mixed_intent_clarification(user_message: str, planner: Planner) -> RequestClassification | None:
    normalized = _normalize(user_message)
    segments = _split_intent_segments(normalized)
    if len(segments) < 2:
        return None

    decisions = [planner.decide_action(segment) for segment in segments]
    has_action = any(decision.intent_type == "EXECUTE_TOOL" for decision in decisions)
    has_non_action_intent = any(
        decision.intent_type != "EXECUTE_TOOL" or _looks_like_weather_request(segment)
        for decision, segment in zip(decisions, segments, strict=True)
    )
    if not has_action or not has_non_action_intent:
        return None

    return RequestClassification(
        RequestCategory.ask_clarification,
        "This message contains multiple intents. Please split it into separate requests or confirm the multi-step task.",
        missing_input="intent",
    )


def _browser_tool_sequence(user_message: str) -> list[ToolProposal]:
    normalized = _normalize(user_message)
    url = _extract_url(user_message)
    if url and _has_browser_open_intent(normalized) and _has_browser_read_intent(normalized):
        return [
            ToolProposal("browser_open", {"url": url}),
            ToolProposal("browser_read_page", {}),
        ]
    return []


def _browser_tool_proposal(user_message: str) -> tuple[ToolProposal | None, str | None, str]:
    normalized = _normalize(user_message)
    if _is_browser_instructional_question(normalized):
        return None, None, ""
    url = _extract_url(user_message)

    if url and _has_browser_open_intent(normalized):
        return ToolProposal("browser_open", {"url": url}), None, "User asked to open a URL in the controlled browser."

    if _has_browser_open_intent(normalized) and any(term in normalized for term in ("website", "url", "page")):
        return None, "url", "Which URL should I open?"

    if _has_browser_read_intent(normalized):
        return ToolProposal("browser_read_page", {}), None, "User asked to read the active browser page."

    if "links" in normalized and ("page" in normalized or "browser" in normalized or "website" in normalized):
        return ToolProposal("browser_extract_links", {}), None, "User asked to extract links from the active browser page."

    if ("forms" in normalized or "form fields" in normalized or "fields" in normalized) and (
        "page" in normalized or "browser" in normalized or "website" in normalized or "form" in normalized
    ):
        return ToolProposal("browser_extract_forms", {}), None, "User asked to extract forms from the active browser page."

    if "screenshot" in normalized and ("browser" in normalized or "page" in normalized or "website" in normalized):
        return ToolProposal("browser_screenshot", {}), None, "User asked to capture the controlled browser page."

    if _has_browser_search_intent(normalized):
        return ToolProposal("browser_search", {"query": _browser_search_query(user_message)}), None, "User asked to search the web in the controlled browser."

    click_match = re.search(r"\bclick\s+((?:link|button)_\d{3})\b", normalized)
    if click_match:
        return ToolProposal("browser_click", {"element_id": click_match.group(1)}), None, "User asked to click a known browser element."

    fill_match = re.search(r"\bfill\s+((?:input|textarea|select)_\d{3})\s+(?:with|as)\s+(.+)$", normalized)
    if fill_match:
        return (
            ToolProposal("browser_fill", {"element_id": fill_match.group(1), "value": fill_match.group(2).strip(" .")}),
            None,
            "User asked to fill a known browser field.",
        )

    if re.search(r"\bsubmit\s+(?:this\s+)?form\b", normalized):
        return ToolProposal("browser_submit_form", {"form_id": "form_001"}), None, "User asked to submit the active browser form."

    return None, None, ""


def _has_browser_open_intent(normalized: str) -> bool:
    return any(phrase in normalized for phrase in ("open http", "go to http", "navigate to http", "open file://", "open website", "open url", "go to "))


def _has_browser_read_intent(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "read this page",
            "read the page",
            "read current page",
            "summarize this page",
            "summarize the page",
            "what is on this page",
        )
    )


def _extract_url(user_message: str) -> str | None:
    match = re.search(r"\b(?:https?://|file://)[^\s]+", user_message, flags=re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,)")
    return None


def _has_browser_search_intent(normalized: str) -> bool:
    if any(phrase in normalized for phrase in ("search my files", "find file", "find my file")):
        return False
    return (
        normalized.startswith("search web for ")
        or normalized.startswith("search the web for ")
        or normalized.startswith("find official website")
        or normalized.startswith("find reservation page")
        or normalized.startswith("find website")
    )


def _browser_search_query(user_message: str) -> str:
    normalized = _normalize(user_message)
    for prefix in ("search the web for ", "search web for ", "find "):
        if normalized.startswith(prefix):
            return user_message[len(prefix):].strip(" ?.") or user_message.strip()
    return user_message.strip()


def _looks_like_negated_browser_action(normalized: str) -> bool:
    return re.search(r"\b(don't|dont|do not|never|stop)\s+(submit|click|fill|book|reserve|order|pay)\b", normalized) is not None


def _is_browser_instructional_question(normalized: str) -> bool:
    return re.search(r"^(how do i|how can i|what happens if|what if|can i|should i)\b.*\b(submit|click|fill|open|book|reserve|order|pay)\b", normalized) is not None


def _split_intent_segments(normalized: str) -> list[str]:
    parts = re.split(r"\s+(?:and then|then|and)\s+", normalized)
    return [part.strip(" .,;") for part in parts if part.strip(" .,;")]


def _looks_like_selected_text_request(normalized: str) -> bool:
    if any(phrase in normalized for phrase in ("selected word", "selected text", "highlighted word", "highlighted text", "selection")):
        return True

    selected_verbs = (
        "what text did i mark",
        "what word did i mark",
        "what text did i highlight",
        "what word did i highlight",
        "what did i select",
        "what have i selected",
        "what i selected",
        "what do i have selected",
        "what word do i have selected",
        "what text do i have selected",
        "what text did i select",
        "what text have i selected",
        "what text i have selected",
        "what text i selected",
        "what did i highlight",
        "what have i highlighted",
        "what i highlighted",
    )
    if any(phrase in normalized for phrase in selected_verbs):
        return True

    return bool(re.search(r"\b(selected|highlighted)\b.*\b(chrome|google|browser|page|window|app|text|word)\b", normalized))


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
    # Pick the last plausible location preposition so "weather at 7pm in Warsaw" resolves to Warsaw.
    text = user_message.strip()
    candidates: list[str] = []
    for match in re.finditer(r"\b(in|for|near|at)\s+", text, flags=re.IGNORECASE):
        candidate = text[match.end():]
        cleaned = _clean_weather_location_candidate(candidate)
        if cleaned:
            candidates.append(cleaned)

    return candidates[-1] if candidates else None


def _clean_weather_location_candidate(candidate: str) -> str | None:
    value = candidate.strip(" ?.,")
    value = re.sub(r"\b(on\s+)?\d{4}-\d{2}-\d{2}\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(today|tomorrow|tonight|now|currently)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(this\s+)?(weekend|week|morning|afternoon|evening|night)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)?\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(weather|forecast|temperature|rain|raining)\b", "", value, flags=re.IGNORECASE)
    value = " ".join(value.split()).strip(" ?.,")
    if not value or value.lower() in {"the", "there", "outside", "today", "tomorrow"}:
        return None
    return value


def _looks_like_weather_request(normalized: str) -> bool:
    return any(keyword in normalized for keyword in ("weather", "forecast", "temperature", "rain", "raining"))


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
    return canonicalize_user_text(value)
