from app.schemas.proposed_tools import ProposedToolDraft


def propose_missing_tool(user_message: str) -> ProposedToolDraft | None:
    text = user_message.lower()

    if "find" in text or "search" in text or "file" in text:
        return ProposedToolDraft(
            name="search_files",
            description="Search indexed local folders for files matching a query.",
            reason="The user asked to find a file, but no implemented file-search tool exists.",
            risk_level="READ",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "roots": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "matches": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["matches"],
                "additionalProperties": False,
            },
        )

    if "screen" in text or "look at" in text or "read this" in text:
        return ProposedToolDraft(
            name="capture_active_window",
            description="Capture the active window for visual question answering.",
            reason="The user asked the assistant to inspect screen content.",
            risk_level="READ",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"image_id": {"type": "string"}},
                "required": ["image_id"],
                "additionalProperties": False,
            },
        )

    return None
