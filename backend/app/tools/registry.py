from typing import Any

from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for definition in _default_tool_definitions():
        registry.register(definition)
    return registry


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _success_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
        },
        ["status", "message"],
    )


def _default_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="launch_app",
            description="Open an installed application by a safe app name or executable alias.",
            input_schema=_object_schema({"app_name": {"type": "string"}}, ["app_name"]),
            output_schema=_success_output_schema(),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=10,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="search_files",
            description="Search local indexed locations for files matching a user query.",
            input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
            output_schema=_object_schema({"matches": {"type": "array", "items": {"type": "string"}}}),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=20,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="open_file",
            description="Open a selected file after the backend has resolved the path.",
            input_schema=_object_schema({"path": {"type": "string"}}, ["path"]),
            output_schema=_success_output_schema(),
            risk_level=RiskLevel.medium,
            confirmation_policy=ConfirmationPolicy.before_execute,
            timeout_seconds=10,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="capture_active_window",
            description="Capture the currently active window for visual question answering.",
            input_schema=_object_schema({}),
            output_schema=_object_schema({"image_id": {"type": "string"}}),
            risk_level=RiskLevel.medium,
            confirmation_policy=ConfirmationPolicy.before_execute,
            timeout_seconds=10,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="get_accessibility_tree",
            description="Read the accessibility tree for the focused window.",
            input_schema=_object_schema({}),
            output_schema=_object_schema({"tree": {"type": "object"}}),
            risk_level=RiskLevel.medium,
            confirmation_policy=ConfirmationPolicy.before_execute,
            timeout_seconds=10,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="browser_open",
            description="Open a browser page by URL.",
            input_schema=_object_schema({"url": {"type": "string"}}, ["url"]),
            output_schema=_success_output_schema(),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=10,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="browser_search",
            description="Search the web in the browser and return candidate results.",
            input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
            output_schema=_object_schema({"results": {"type": "array", "items": {"type": "object"}}}),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=30,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="browser_fill",
            description="Fill fields in a browser page up to a draft or confirmation boundary.",
            input_schema=_object_schema(
                {
                    "selector": {"type": "string"},
                    "value": {"type": "string"},
                },
                ["selector", "value"],
            ),
            output_schema=_success_output_schema(),
            risk_level=RiskLevel.high,
            confirmation_policy=ConfirmationPolicy.before_execute,
            timeout_seconds=15,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="speak_text",
            description="Speak assistant text through the configured local voice.",
            input_schema=_object_schema({"text": {"type": "string"}}, ["text"]),
            output_schema=_success_output_schema(),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=20,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        ToolDefinition(
            name="transcribe_audio",
            description="Transcribe push-to-talk audio into text.",
            input_schema=_object_schema({"audio_id": {"type": "string"}}, ["audio_id"]),
            output_schema=_object_schema({"text": {"type": "string"}}, ["text"]),
            risk_level=RiskLevel.low,
            confirmation_policy=ConfirmationPolicy.none,
            timeout_seconds=30,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
    ]
