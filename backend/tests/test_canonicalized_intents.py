import pytest

from app.assistant.request_classifier import RequestCategory, classify_request
from app.assistant.text_canonicalization import canonicalize_user_text


@pytest.mark.parametrize(
    ("message", "expected_tool"),
    [
        ("what word i have in clip board", "get_clipboard_text"),
        ("what is in my copy buffer", "get_clipboard_text"),
        ("what did I copy", "get_clipboard_text"),
        ("what is the answer to what I highlighted", "get_selected_text"),
        ("what is the answer to the highlighted question", "get_selected_text"),
        ("answer selected question", "get_selected_text"),
        ("what is the highlighted question", "get_selected_text"),
        ("what did I select", "get_selected_text"),
        ("what have I highlighted", "get_selected_text"),
        ("what does selected work mean", "get_selected_text"),
        ("what text did I mark", "get_selected_text"),
        ("what word did I highlight", "get_selected_text"),
    ],
)
def test_fuzzy_context_phrases_route_to_local_context(message: str, expected_tool: str) -> None:
    classification = classify_request(message)

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == expected_tool


def test_canonicalization_does_not_globally_replace_work_with_word() -> None:
    assert canonicalize_user_text("how does this work?") == "how does this work?"
    assert canonicalize_user_text("what does selected work mean?") == "what does selected word mean?"
