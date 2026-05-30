import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import logging
from typing import Any

from app.browser.session_manager import BrowserSessionManager
from app.context.window_tracker import WindowTracker
from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition, ToolResult, ToolStatus
from app.tools.browser_tools import build_browser_tool_executors
from app.tools.context_tools import SelectedTextCaptureStrategy, build_context_tool_executors
from app.tools.system_tools import get_current_time, launch_app
from app.tools.weather_tools import get_weather

LOGGER = logging.getLogger("app.tools.registry")
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
        LOGGER.info(
            "tool registered | name=%s | status=%s | executable=%s",
            metadata.name,
            metadata.status.value,
            executor is not None,
        )

    def get(self, name: str) -> ToolDefinition | None:
        runtime = self._tools.get(name)
        return runtime.metadata if runtime else None

    def get_runtime(self, name: str) -> ToolRuntimeDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return [runtime.metadata for runtime in self._tools.values()]

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> str | None:
        runtime = self._tools.get(name)
        if runtime is None:
            return f"Unknown tool: {name}"
        return self._validate_required_arguments(runtime.metadata, arguments)

    async def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        LOGGER.info("tool execution requested | name=%s | arguments=%s", name, _json_for_log(arguments))
        runtime = self._tools.get(name)
        if runtime is None:
            LOGGER.warning("tool execution rejected | name=%s | reason=unknown_tool", name)
            return ToolResult(tool=name, status="failed", error=f"Unknown tool: {name}")

        metadata = runtime.metadata
        if metadata.status != ToolStatus.implemented or runtime.executor is None:
            LOGGER.warning(
                "tool execution rejected | name=%s | status=%s | executable=%s",
                name,
                metadata.status.value,
                runtime.executor is not None,
            )
            return ToolResult(tool=name, status="failed", error=f"Tool is not implemented: {name}")

        validation_error = self._validate_required_arguments(metadata, arguments)
        if validation_error is not None:
            LOGGER.warning("tool validation failed | name=%s | error=%s", name, validation_error)
            return ToolResult(tool=name, status="failed", error=validation_error)

        last_result: ToolResult | None = None
        for attempt in range(1, metadata.retry_policy.max_attempts + 1):
            try:
                LOGGER.info("tool attempt started | name=%s | attempt=%s/%s", name, attempt, metadata.retry_policy.max_attempts)
                result = await asyncio.wait_for(runtime.executor(arguments), timeout=metadata.timeout_seconds)
                if result.status == "success":
                    LOGGER.info("tool attempt succeeded | name=%s | attempt=%s | output=%s", name, attempt, _json_for_log(result.output))
                else:
                    LOGGER.warning("tool attempt failed | name=%s | attempt=%s | error=%s", name, attempt, result.error)
                return result
            except TimeoutError:
                last_result = ToolResult(tool=name, status="failed", error=f"Tool timed out after {metadata.timeout_seconds} seconds.")
                LOGGER.warning("tool attempt timed out | name=%s | attempt=%s | timeout_seconds=%s", name, attempt, metadata.timeout_seconds)
            except Exception as exc:
                last_result = ToolResult(tool=name, status="failed", error=f"Tool raised an exception: {exc}")
                LOGGER.exception("tool attempt raised exception | name=%s | attempt=%s", name, attempt)
                return last_result

            if metadata.retry_policy.backoff_seconds > 0:
                await asyncio.sleep(metadata.retry_policy.backoff_seconds)

        LOGGER.warning("tool execution exhausted retries | name=%s", name)
        return last_result or ToolResult(tool=name, status="failed", error="Tool execution failed.")

    @staticmethod
    def _validate_required_arguments(metadata: ToolDefinition, arguments: dict[str, object]) -> str | None:
        if metadata.input_schema.get("additionalProperties") is False:
            allowed = metadata.input_schema.get("properties", {})
            if isinstance(allowed, dict):
                extra = sorted(set(arguments) - set(allowed))
                if extra:
                    return f"Unexpected argument(s): {', '.join(extra)}"

        required = metadata.input_schema.get("required", [])
        if not isinstance(required, list):
            return None

        missing = [field for field in required if isinstance(field, str) and field not in arguments]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"

        properties = metadata.input_schema.get("properties", {})
        if isinstance(properties, dict):
            for name, schema in properties.items():
                if name not in arguments or not isinstance(schema, dict):
                    continue
                expected_type = schema.get("type")
                value = arguments[name]
                if expected_type == "string" and not isinstance(value, str):
                    return f"Argument '{name}' must be a string."
                if expected_type == "object" and not isinstance(value, dict):
                    return f"Argument '{name}' must be an object."
                if expected_type == "array" and not isinstance(value, list):
                    return f"Argument '{name}' must be an array."
        return None


def build_default_registry(
    window_tracker: WindowTracker | None = None,
    selected_text_strategy: SelectedTextCaptureStrategy | None = None,
    browser_manager: BrowserSessionManager | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    window_tracker = window_tracker or WindowTracker()
    browser_manager = browser_manager or BrowserSessionManager()
    context_executors = build_context_tool_executors(window_tracker, selected_text_strategy)
    browser_executors = build_browser_tool_executors(browser_manager)
    for definition, executor in _default_tool_runtimes(context_executors, browser_executors):
        registry.register(definition, executor)
    registry.register(_list_available_tools_definition(), _build_list_available_tools_executor(registry))
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


def _list_available_tools_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "tools": {"type": "array", "items": {"type": "object"}},
        },
        ["status", "message", "tools"],
    )


def _time_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "iso_time": {"type": "string"},
            "timezone": {"type": "string"},
        },
        ["status", "message", "iso_time", "timezone"],
    )


def _weather_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "location": {"type": "string"},
            "date": {"type": "string"},
            "temperature": {"type": "object"},
            "condition": {"type": "string"},
            "precipitation_probability": {"type": "number"},
            "wind": {"type": "object"},
            "source": {"type": "string"},
            "source_urls": {"type": "object"},
            "resolved_coordinates": {"type": "object"},
            "timezone": {"type": "string"},
            "units": {"type": "object"},
            "forecast_generation_time_ms": {"type": "number"},
            "geocoding": {"type": "object"},
        },
        [
            "status",
            "message",
            "location",
            "temperature",
            "condition",
            "precipitation_probability",
            "wind",
            "source",
            "source_urls",
            "resolved_coordinates",
            "timezone",
            "units",
            "geocoding",
        ],
    )


def _window_info_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "window": {"type": "object"},
            "source": {"type": "string"},
        },
        ["status", "message", "window", "source"],
    )


def _clipboard_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "text": {"type": "string"},
            "length": {"type": "number"},
            "source": {"type": "string"},
            "captured_at": {"type": "string"},
        },
        ["status", "message", "text", "length", "source", "captured_at"],
    )


def _selected_text_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "text": {"type": "string"},
            "length": {"type": "number"},
            "method": {"type": "string"},
            "context_window": {"type": "object"},
            "restored_clipboard": {"type": "boolean"},
            "metadata": {"type": "object"},
            "captured_at": {"type": "string"},
        },
        ["status", "message", "text", "length", "method", "context_window", "restored_clipboard"],
    )


def _browser_snapshot_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "text_preview": {"type": "string"},
            "text_length": {"type": "number"},
            "links_count": {"type": "number"},
            "forms_count": {"type": "number"},
            "snapshot": {"type": "object"},
        },
        ["status", "message", "url", "title", "snapshot"],
    )


def _browser_links_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "links": {"type": "array", "items": {"type": "object"}},
            "snapshot": {"type": "object"},
        },
        ["status", "message", "links"],
    )


def _browser_forms_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "forms": {"type": "array", "items": {"type": "object"}},
            "snapshot": {"type": "object"},
        },
        ["status", "message", "forms"],
    )


def _browser_screenshot_output_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "status": {"type": "string"},
            "message": {"type": "string"},
            "screenshot": {"type": "object"},
            "snapshot": {"type": "object"},
        },
        ["status", "message", "screenshot"],
    )


def _list_available_tools_definition() -> ToolDefinition:
    return ToolDefinition(
        name="list_available_tools",
        description="List registered assistant tools and their implementation status.",
        status=ToolStatus.implemented,
        input_schema=_object_schema({}),
        output_schema=_list_available_tools_output_schema(),
        risk_level=RiskLevel.read,
        confirmation_policy=ConfirmationPolicy.none,
        timeout_seconds=5,
        retry_policy=RetryPolicy(max_attempts=1),
    )


def _build_list_available_tools_executor(registry: ToolRegistry) -> ToolExecutor:
    async def list_available_tools(arguments: dict[str, object]) -> ToolResult:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "status": tool.status.value,
                "risk_level": tool.risk_level.value,
                "confirmation_policy": tool.confirmation_policy.value,
            }
            for tool in registry.list_tools()
        ]
        implemented = [tool["name"] for tool in tools if tool["status"] == ToolStatus.implemented.value]
        return ToolResult(
            tool="list_available_tools",
            status="success",
            output={
                "status": "success",
                "message": f"Available implemented tools: {', '.join(implemented)}.",
                "tools": tools,
            },
        )

    return list_available_tools


def _default_tool_runtimes(
    context_executors: dict[str, ToolExecutor],
    browser_executors: dict[str, ToolExecutor],
) -> list[tuple[ToolDefinition, ToolExecutor | None]]:
    return [
        (
            ToolDefinition(
                name="launch_app",
                description="Open a whitelisted Windows application.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"app_name": {"type": "string"}}, ["app_name"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.low_write,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            launch_app,
        ),
        (
            ToolDefinition(
                name="get_current_time",
                description="Return the current local date and time.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_time_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=5,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            get_current_time,
        ),
        (
            ToolDefinition(
                name="get_weather",
                description="Return current or forecast weather for a named location using Open-Meteo.",
                status=ToolStatus.implemented,
                input_schema=_object_schema(
                    {
                        "location": {"type": "string"},
                        "date": {"type": "string", "description": "today, tomorrow, or YYYY-MM-DD"},
                    },
                    ["location", "date"],
                ),
                output_schema=_weather_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            get_weather,
        ),
        (
            ToolDefinition(
                name="get_foreground_window_info",
                description="Return the actual current foreground window, even if it is Vix.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_window_info_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=5,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            context_executors["get_foreground_window_info"],
        ),
        (
            ToolDefinition(
                name="get_context_window_info",
                description="Return the last non-Vix context window for user-perspective references like this window or current app.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_window_info_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=5,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            context_executors["get_context_window_info"],
        ),
        (
            ToolDefinition(
                name="get_clipboard_text",
                description="Read the current text clipboard. Privacy-sensitive read.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_clipboard_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=5,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            context_executors["get_clipboard_text"],
        ),
        (
            ToolDefinition(
                name="get_selected_text",
                description="Capture selected text from the context window using temporary focus and Ctrl+C. Privacy-sensitive read.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_selected_text_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=8,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            context_executors["get_selected_text"],
        ),
        (
            ToolDefinition(
                name="search_files",
                description="Search local indexed locations for files matching a user query.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
                output_schema=_object_schema({"matches": {"type": "array", "items": {"type": "string"}}}),
                risk_level=RiskLevel.read,
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
                risk_level=RiskLevel.medium_write,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="create_file",
                description="Create a local file after preview and permission.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.medium_write,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="send_message",
                description="Send a drafted message to a recipient after explicit approval.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"recipient": {"type": "string"}, "message": {"type": "string"}}, ["recipient", "message"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.high_risk,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="pay_for_order",
                description="Submit payment for a prepared order after explicit approval.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"order_id": {"type": "string"}, "amount": {"type": "string"}}, ["order_id", "amount"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.high_risk,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=15,
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
                risk_level=RiskLevel.read,
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
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=10,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
        (
            ToolDefinition(
                name="browser_open",
                description="Open a URL in Vix's isolated controlled Playwright browser session.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"url": {"type": "string"}}, ["url"]),
                output_schema=_browser_snapshot_output_schema(),
                risk_level=RiskLevel.low_write,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=20,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_open"],
        ),
        (
            ToolDefinition(
                name="browser_read_page",
                description="Read the active controlled browser page and return a structured snapshot.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_browser_snapshot_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_read_page"],
        ),
        (
            ToolDefinition(
                name="browser_extract_links",
                description="Extract links from the active controlled browser page using internal element IDs.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_browser_links_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_extract_links"],
        ),
        (
            ToolDefinition(
                name="browser_extract_forms",
                description="Extract forms and fields from the active controlled browser page using internal element IDs.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_browser_forms_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_extract_forms"],
        ),
        (
            ToolDefinition(
                name="browser_screenshot",
                description="Capture a screenshot of the controlled browser page. Privacy-sensitive read.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({}),
                output_schema=_browser_screenshot_output_schema(),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_screenshot"],
        ),
        (
            ToolDefinition(
                name="browser_click",
                description="Click a known browser element ID. Stage 5 supports safe navigation clicks only.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"element_id": {"type": "string"}}, ["element_id"]),
                output_schema=_browser_snapshot_output_schema(),
                risk_level=RiskLevel.low_write,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_click"],
        ),
        (
            ToolDefinition(
                name="browser_fill",
                description="Fill a known browser field ID as a draft. This never submits the form.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"element_id": {"type": "string"}, "value": {"type": "string"}}, ["element_id", "value"]),
                output_schema=_browser_snapshot_output_schema(),
                risk_level=RiskLevel.medium_write,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_fill"],
        ),
        (
            ToolDefinition(
                name="browser_submit_form",
                description="Submit a known browser form ID. Always requires explicit permission.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"form_id": {"type": "string"}}, ["form_id"]),
                output_schema=_browser_snapshot_output_schema(),
                risk_level=RiskLevel.high_risk,
                confirmation_policy=ConfirmationPolicy.before_execute,
                timeout_seconds=15,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_submit_form"],
        ),
        (
            ToolDefinition(
                name="browser_search",
                description="Open a search results page in the controlled browser and return candidate result links.",
                status=ToolStatus.implemented,
                input_schema=_object_schema({"query": {"type": "string"}}, ["query"]),
                output_schema=_object_schema(
                    {
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                        "query": {"type": "string"},
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "results": {"type": "array", "items": {"type": "object"}},
                        "snapshot": {"type": "object"},
                    },
                    ["status", "message", "query", "results"],
                ),
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=30,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            browser_executors["browser_search"],
        ),
        (
            ToolDefinition(
                name="speak_text",
                description="Speak assistant text through the configured local voice.",
                status=ToolStatus.planned,
                input_schema=_object_schema({"text": {"type": "string"}}, ["text"]),
                output_schema=_success_output_schema(),
                risk_level=RiskLevel.low_write,
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
                risk_level=RiskLevel.read,
                confirmation_policy=ConfirmationPolicy.none,
                timeout_seconds=30,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            None,
        ),
    ]


def _json_for_log(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
