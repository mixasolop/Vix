import re


def canonicalize_user_text(text: str) -> str:
    """Normalize user text for deterministic intent detection only."""

    value = text.lower()
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", value)
    value = re.sub(r"\bclip[\s-]*board\b", "clipboard", value)
    value = re.sub(r"\bclipbord\b", "clipboard", value)
    value = re.sub(r"\bselcted\s+word\b", "selected word", value)
    value = re.sub(r"\bselect\s+word\b", "selected word", value)
    value = re.sub(r"\bhighlight\s+word\b", "highlighted word", value)
    value = re.sub(r"\bhighlight\s+text\b", "highlighted text", value)

    # Only treat "work" as "word" inside clear selection phrases. Do not
    # globally replace it, because "work" is a valid word in ordinary requests.
    selection_work_patterns = (
        (r"\bselected\s+work\b", "selected word"),
        (r"\bthe\s+selected\s+work\b", "the selected word"),
        (r"\bwhat\s+work\s+do\s+i\s+have\s+selected\b", "what word do i have selected"),
        (r"\bwhat\s+does\s+selected\s+work\s+mean\b", "what does selected word mean"),
    )
    for pattern, replacement in selection_work_patterns:
        value = re.sub(pattern, replacement, value)

    value = re.sub(r"\s+", " ", value)
    return value.strip()
