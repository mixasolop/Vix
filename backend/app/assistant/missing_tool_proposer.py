from app.schemas.proposed_tools import ProposedToolDraft


def propose_missing_tool(user_message: str) -> ProposedToolDraft | None:
    text = user_message.lower()

    if any(keyword in text for keyword in ("find", "search my files", "search file")):
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

    if any(keyword in text for keyword in ("screen", "screenshot", "look at", "read this")):
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

    if "clipboard" in text or "selected text" in text:
        return ProposedToolDraft(
            name="read_clipboard_or_selection",
            description="Read clipboard or selected text content after user-controlled context capture.",
            reason="The user asked Vix to use clipboard or selected text context, but no implemented context-capture tool exists.",
            risk_level="READ",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "source": {"type": "string"}},
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        )

    if any(keyword in text for keyword in ("send", "message", "email", "telegram", "discord")):
        return ProposedToolDraft(
            name="send_message",
            description="Draft and send a message only after explicit human approval.",
            reason="The user asked Vix to send or draft an external message, but messaging tools are not implemented yet.",
            risk_level="HIGH_RISK",
            input_schema={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["recipient", "message"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "message_id": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )

    if any(keyword in text for keyword in ("browser", "website", "web page", "search web", "google")):
        return ProposedToolDraft(
            name="browser_search_or_open",
            description="Open or search the browser through controlled browser tools.",
            reason="The user asked for browser/web interaction, but browser tools are not implemented yet.",
            risk_level="LOW_WRITE",
            input_schema={
                "type": "object",
                "properties": {"query_or_url": {"type": "string"}},
                "required": ["query_or_url"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "results": {"type": "array", "items": {"type": "object"}}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )

    if any(keyword in text for keyword in ("folder", "create file", "write file", "edit file", "copy file", "move file", "rename file", "delete file", "open file")):
        return ProposedToolDraft(
            name="manage_files",
            description="Create, edit, move, rename, delete, or open local files after policy review.",
            reason="The user asked for local file management, but the needed file operation is not implemented yet.",
            risk_level="MEDIUM_WRITE",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "path": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )

    if any(keyword in text for keyword in ("calendar", "reminder", "schedule")):
        return ProposedToolDraft(
            name="manage_calendar_or_reminders",
            description="Create reminders or calendar events after explicit review.",
            reason="The user asked for reminder/calendar management, but no calendar tool exists yet.",
            risk_level="MEDIUM_WRITE",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "event_id": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )

    if text.startswith(("open ", "launch ", "start ", "run ")):
        return ProposedToolDraft(
            name="launch_additional_app",
            description="Launch a reviewed desktop application that is not currently in the safe app whitelist.",
            reason="The user asked to launch an app that is not mapped to the current launch_app whitelist.",
            risk_level="LOW_WRITE",
            input_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "message": {"type": "string"}},
                "required": ["status", "message"],
                "additionalProperties": False,
            },
        )

    if any(keyword in text for keyword in ("automate", "control", "manage", "setting", "settings", "device", "smart home", "mixer", "music")):
        return ProposedToolDraft(
            name="desktop_control_action",
            description="Perform a reviewed desktop or device-control action through a future safe tool.",
            reason="The user asked for a control/automation capability that is not implemented yet.",
            risk_level="MEDIUM_WRITE",
            input_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}, "message": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        )

    return ProposedToolDraft(
        name="new_assistant_capability",
        description="A new assistant capability discovered from an unsupported user request.",
        reason="The assistant could not map this action-like request to an implemented tool.",
        risk_level="MEDIUM_WRITE",
        input_schema={
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}, "message": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )
