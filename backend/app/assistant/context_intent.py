from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from app.assistant.text_canonicalization import canonicalize_user_text


class ContextSource(StrEnum):
    selected_text = "selected_text"
    clipboard = "clipboard"
    context_window = "context_window"
    foreground_window = "foreground_window"


class ContextOperation(StrEnum):
    report = "report"
    explain = "explain"
    define = "define"
    answer_question = "answer_question"
    summarize = "summarize"
    unknown = "unknown"


@dataclass(frozen=True)
class ContextIntent:
    source: ContextSource
    operation: ContextOperation
    tool_name: str
    arguments: dict[str, object]
    reason: str
    confidence: float


def resolve_context_intent(user_message: str) -> ContextIntent | None:
    """Resolve references to local user context before the normal answer path.

    This intentionally treats selected/highlighted/clipboard language as a hard
    dependency on a context tool. The LLM may answer after capture, but it should
    not guess what the user selected or copied.
    """

    normalized = canonicalize_user_text(user_message)
    if not normalized:
        return None

    clipboard_intent = _clipboard_intent(normalized)
    if clipboard_intent is not None:
        return clipboard_intent

    selected_intent = _selected_text_intent(normalized)
    if selected_intent is not None:
        return selected_intent

    window_intent = _window_intent(normalized)
    if window_intent is not None:
        return window_intent

    return None


def _clipboard_intent(normalized: str) -> ContextIntent | None:
    if not _mentions_clipboard(normalized):
        return None

    operation = _operation_for_clipboard(normalized)
    return ContextIntent(
        source=ContextSource.clipboard,
        operation=operation,
        tool_name="get_clipboard_text",
        arguments={},
        reason=_reason_for(ContextSource.clipboard, operation),
        confidence=0.96,
    )


def _selected_text_intent(normalized: str) -> ContextIntent | None:
    if not _mentions_selected_or_highlighted_text(normalized):
        return None

    operation = _operation_for_selected_text(normalized)
    return ContextIntent(
        source=ContextSource.selected_text,
        operation=operation,
        tool_name="get_selected_text",
        arguments={},
        reason=_reason_for(ContextSource.selected_text, operation),
        confidence=0.96,
    )


def _window_intent(normalized: str) -> ContextIntent | None:
    if "foreground window" in normalized or "technical foreground" in normalized:
        return ContextIntent(
            source=ContextSource.foreground_window,
            operation=ContextOperation.report,
            tool_name="get_foreground_window_info",
            arguments={},
            reason="User asked for the technical foreground window.",
            confidence=0.95,
        )

    if any(
        phrase in normalized
        for phrase in (
            "what app am i using",
            "what app was i using",
            "which app am i using",
            "which app was i using",
            "context window",
            "previous window",
            "this window",
            "current window",
        )
    ):
        return ContextIntent(
            source=ContextSource.context_window,
            operation=ContextOperation.report,
            tool_name="get_context_window_info",
            arguments={},
            reason="User asked for the last non-Vix context window.",
            confidence=0.94,
        )

    return None


def _mentions_clipboard(normalized: str) -> bool:
    return (
        "clipboard" in normalized
        or "copy buffer" in normalized
        or "copied text" in normalized
        or re.search(r"\bwhat(?:\s+text)?\s+did\s+i\s+copy\b", normalized) is not None
        or re.search(r"\bwhat\s+(?:is|text)\s+.*\bcopy\b", normalized) is not None
    )


def _mentions_selected_or_highlighted_text(normalized: str) -> bool:
    if any(term in normalized for term in ("highlighted", "selected", "marked", "selection")):
        return True

    if re.search(r"\bwhat\s+(?:did|have)\s+i\s+(?:highlight|select|mark)\b", normalized):
        return True

    if re.search(r"\bwhat\s+(?:text|word)\s+did\s+i\s+(?:highlight|select|mark)\b", normalized):
        return True

    if re.search(r"\banswer\s+(?:the\s+)?(?:highlight|select|mark)", normalized):
        return True

    return False


def _operation_for_clipboard(normalized: str) -> ContextOperation:
    if "summarize" in normalized or "summary" in normalized:
        return ContextOperation.summarize
    if "explain" in normalized:
        return ContextOperation.explain
    if "mean" in normalized or "meaning" in normalized or "define" in normalized:
        return ContextOperation.define
    if (
        "what did i copy" in normalized
        or "what text did i copy" in normalized
        or "what is in" in normalized
        or "what word" in normalized
        or "what text" in normalized
    ):
        return ContextOperation.report
    return ContextOperation.unknown


def _operation_for_selected_text(normalized: str) -> ContextOperation:
    if (
        "answer" in normalized
        and any(term in normalized for term in ("highlight", "selected", "select", "marked", "mark", "question"))
    ):
        return ContextOperation.answer_question

    if "summarize" in normalized or "summary" in normalized:
        return ContextOperation.summarize

    if "mean" in normalized or "meaning" in normalized or "define" in normalized:
        if "word" in normalized:
            return ContextOperation.define
        return ContextOperation.explain

    if "explain" in normalized:
        return ContextOperation.explain

    if (
        re.search(r"\bwhat\s+(?:did|have)\s+i\s+(?:highlight|select|mark)", normalized)
        or re.search(r"\bwhat\s+(?:text|word)\s+(?:did|have)\s+i\s+(?:highlight|select|mark)", normalized)
        or "what is highlighted" in normalized
        or "what is selected" in normalized
        or "what is the highlighted" in normalized
        or "what is the selected" in normalized
    ):
        return ContextOperation.report

    return ContextOperation.unknown


def _reason_for(source: ContextSource, operation: ContextOperation) -> str:
    source_label = {
        ContextSource.selected_text: "selected or highlighted text",
        ContextSource.clipboard: "clipboard text",
        ContextSource.context_window: "context window",
        ContextSource.foreground_window: "foreground window",
    }[source]
    operation_label = {
        ContextOperation.report: "report it",
        ContextOperation.explain: "explain it",
        ContextOperation.define: "define it",
        ContextOperation.answer_question: "answer it as a question",
        ContextOperation.summarize: "summarize it",
        ContextOperation.unknown: "use it as local context",
    }[operation]
    return f"User referenced {source_label}; capture it first, then {operation_label}."
