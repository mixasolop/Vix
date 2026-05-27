import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition, ToolResult, ToolStatus
from app.tools.system_tools import launch_app

ToolExecutor = Callable[[dict[str, object]], Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolRuntimeDefinition:
    metadata: ToolDefinition
    executor: ToolExecutor | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolRuntimeDefinition] = {}

    def register(self, metadata: ToolDefinition, executor: ToolExecutor | None = None) -> None:
        self._tools[metadata.name] = ToolRuntimeDefinition(metadata=metadata, executor=executor)

    def get(self, name: str) -> ToolDefinition | None:
        runtime = self._tools.get(name)
        return runtime.metadata if runtime else None

    def get_runtime(self, name: str) -> ToolRuntimeDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return [runtime.metadata for runtime in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        runtime = self._tools.get(name)
        if runtime is None:
            return ToolResult(tool=name, status="failed", error=f"Unknown tool: {name}")

        metadata = runtime.metadata
        if metadata.status != ToolStatus.implemented or runtime.executor is None:
            return ToolResult(tool=name, status="failed", error=f"Tool is not implemented: {name}")

        validation_error = self._validate_required_arguments(metadata, arguments)
        if validation_error is not None:
            return ToolResult(tool=name, status="failed", error=validation_error)

        last_result: ToolResult | None = None
        for _ in range(metadata.retry_policy.max_attempts):
            try:
                return await asyncio.wait_for(runtime.executor(arguments), timeout=metadata.timeout_seconds)
            except TimeoutError:
                last_result = ToolResult(tool=name, status="failed", error=f"Tool timed out after {metadata.timeout_seconds} seconds.")

            if metadata.retry_policy.backoff_seconds > 0:
                await asyncio.sleep(metadata.retry_policy.backoff_seconds)

        return last_result or ToolResult(tool=name, status="failed", error="Tool execution failed.")

    @staticmethod
    def _validate_required_arguments(metadata: ToolDefinition, arguments: dict[str, object]) -> str | None:
        required = metadata.input_schema.get("required", [])
        if not isinstance(required, list):
            return None

        missing = [field for field in required if isinstance(field, str) and field not in arguments]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"
        return None


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for definition, executor in _default_tool_runtimes():
        registry.register(definition, executor)
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


def _default_tool_runtimes() -> list[tuple[ToolDefinition, ToolExecutor | None]]:
    return [
        (
            ToolDefinition(
                name="launch_app",
                description="Open a whitelisted Windows application.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"app_name": {"type": "string"}}, ["app_name"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            launch_app,
        ),
        (
            ToolDefinition(
                name="search_files",
                description="Search local indexed locations for files matching a user query.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
                output_schema=_object_schema({"matches": {"type": "array", "items": {"type": "string"}}}),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=20,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="open_file",
                description="Open a selected file after the backend has resolved the path.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"path": {"type": "string"}}, ["path"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.medium,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="capture_active_window",
                description="Capture the currently active window for visual question answering.",
                status=ToolStatus.planned,
                input_schema=_object_schema({}),
                output_schema=_object_schema({"image_id": {"type": "string"}}),
                risk_level=RiskLevel.medium,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="get_accessibility_tree",
                description="Read the accessibility tree for the focused window.",
                status=ToolStatus.planned,
                input_schema=_object_schema({}),
                output_schema=_object_schema({"tree": {"type": "object"}}),
                risk_level=RiskLevel.medium,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="browser_open",
                description="Open a browser page by URL.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"url": {"type": "string"}}, ["url"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="browser_search",
                description="Search the web in the browser and return candidate results.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
                output_schema=_object_schema({"results": {"type": "array", "items": {"type": "object"}}}),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=30,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="browser_fill",
                description="Fill fields in a browser page up to a draft or confirmation boundary.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"selector": {"type": "string"}, "value": {"type": "string"}}, ["selector", "value"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.high,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="speak_text",
                description="Speak assistant text through the configured local voice.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"text": {"type": "string"}}, ["text"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=20,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="transcribe_audio",
                description="Transcribe push-to-talk audio into text.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"audio_id": {"type": "string"}}, ["audio_id"]),
                output_schema=_object_schema({"text": {"type": "string"}}, ["text"]),
                risk_level=RiskLevel.low,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=30,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
    ]
