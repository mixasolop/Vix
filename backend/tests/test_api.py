import asyncio
from collections.abc import Iterator
from datetime import UTC, date, datetime
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.assistant.request_classifier import RequestCategory, classify_request
from app.assistant.llm_client import DeterministicLLMClient, OpenAILLMClient
from app.assistant.planner import FALLBACK_MESSAGE, Planner
from app.assistant.policy import PolicyEngine
from app.config import AppConfig, load_config_from_file
from app.context.window_tracker import WindowTracker
from app.main import _build_llm_client, _get_ai_status, create_app
from app.schemas.plans import AssistantPlan
from app.schemas.proposed_tools import ProposedToolDraft
from app.schemas.tools import ConfirmationPolicy, RiskLevel, ToolResult, ToolStatus
from app.schemas.window_context import SelectedTextResult, WindowContextSnapshot, WindowInfo
from app.tools import context_tools, weather_tools
from app.tools.context_tools import build_context_tool_executors
from app.tools.registry import build_default_registry


@pytest.fixture()
def client(tmp_path) -> Iterator[TestClient]:
    registry = build_default_registry()
    launch_metadata = registry.get("launch_app")
    assert launch_metadata is not None

    async def fake_launch(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool="launch_app",
            status="success",
            output={"status": "success", "message": f"Test launch for {arguments['app_name']}."},
        )

    registry.register(launch_metadata, fake_launch)
    app = create_app(database_path=tmp_path / "events.sqlite3", registry=registry, config=AppConfig())
    with TestClient(app) as test_client:
        yield test_client


def _window(
    *,
    hwnd: int,
    title: str,
    process_name: str,
    is_vix: bool = False,
) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        process_id=1000 + hwnd,
        process_name=process_name,
        executable_path=f"C:\\Apps\\{process_name}",
        is_vix=is_vix,
        captured_at=datetime.now(UTC),
    )


class FakeWindowTracker:
    def __init__(self, current: WindowInfo | None = None, context: WindowInfo | None = None) -> None:
        self.current = current
        self.context = context
        self.callback = None

    def set_context_updated_callback(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_current_foreground_window(self, refresh: bool = True) -> WindowInfo | None:
        return self.current

    def get_context_window(self, validate_exists: bool = False) -> WindowInfo | None:
        return self.context

    def snapshot(self) -> WindowContextSnapshot:
        return WindowContextSnapshot(
            current_foreground_window=self.current,
            last_non_vix_window=self.context,
            last_context_window=self.context,
            last_context_captured_at=self.context.captured_at if self.context else None,
        )


class FakeSelectedTextStrategy:
    async def capture(self, target: WindowInfo) -> SelectedTextResult:
        return SelectedTextResult(
            status="success",
            text="recursion",
            context_window=target,
            restored_clipboard=True,
        )


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["stage"] == "4.0"


def test_sqlite_runtime_pragmas_and_tools_table(client: TestClient) -> None:
    database = client.app.state.database

    assert database.get_pragma("journal_mode") == "wal"
    assert database.get_pragma("busy_timeout") == "5000"
    assert database.count_rows("tools") >= 1


def test_tools_endpoint_lists_stage_one_contracts(client: TestClient) -> None:
    response = client.get("/tools")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()["tools"]}
    assert tools["launch_app"]["status"] == "implemented"
    assert tools["get_current_time"]["status"] == "implemented"
    assert tools["get_weather"]["status"] == "implemented"
    assert tools["get_foreground_window_info"]["status"] == "implemented"
    assert tools["get_context_window_info"]["status"] == "implemented"
    assert tools["get_clipboard_text"]["status"] == "implemented"
    assert tools["get_selected_text"]["status"] == "implemented"
    assert tools["list_available_tools"]["status"] == "implemented"
    assert tools["create_file"]["risk_level"] == "MEDIUM_WRITE"
    assert tools["send_message"]["risk_level"] == "HIGH_RISK"
    assert tools["pay_for_order"]["risk_level"] == "HIGH_RISK"
    assert tools["transcribe_audio"]["status"] == "planned"


def test_chat_returns_acceptance_only(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "open notepad"})

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["conversation_id"].startswith("sess_")
    assert body["run_id"].startswith("run_")
    assert "assistant_message" not in body
    assert "tool_calls" not in body


@pytest.mark.parametrize(
    ("message", "tool_name", "arguments"),
    [
        ("open notepad", "launch_app", {"app_name": "notepad"}),
        ("launch calculator", "launch_app", {"app_name": "calculator"}),
        ("please run calculator", "launch_app", {"app_name": "calculator"}),
        ("yes, open calculator", "launch_app", {"app_name": "calculator"}),
        ("run calculator please", "launch_app", {"app_name": "calculator"}),
        (
            "please open the thing used for calculating things, named calculator",
            "launch_app",
            {"app_name": "calculator"},
        ),
        ("start paint", "launch_app", {"app_name": "paint"}),
        ("open file explorer", "launch_app", {"app_name": "explorer"}),
        ("what tools do you have", "list_available_tools", {}),
        ("what time is it", "get_current_time", {}),
    ],
)
def test_rule_based_planner_maps_stage_one_commands(message: str, tool_name: str, arguments: dict[str, object]) -> None:
    proposal = Planner().propose_tool_call(message)

    assert proposal is not None
    assert proposal.name == tool_name
    assert proposal.arguments == arguments


def test_rule_based_planner_rejects_unknown_open_targets() -> None:
    assert Planner().propose_tool_call("open spotify") is None


def test_policy_engine_allows_low_risk_implemented_tools() -> None:
    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None

    decision = PolicyEngine().evaluate_tool_call(metadata, {"app_name": "notepad"})

    assert decision.allowed is True
    assert decision.requires_permission is False
    assert decision.blocked is False


def test_policy_engine_requires_permission_for_confirmed_tools() -> None:
    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None
    metadata = metadata.model_copy(update={"confirmation_policy": ConfirmationPolicy.before_execute})

    decision = PolicyEngine().evaluate_tool_call(metadata, {"app_name": "notepad"})

    assert decision.allowed is False
    assert decision.requires_permission is True
    assert decision.blocked is False
    assert "requires confirmation" in decision.reason


def test_policy_engine_blocks_unimplemented_tools() -> None:
    registry = build_default_registry()
    metadata = registry.get("open_file")
    assert metadata is not None

    decision = PolicyEngine().evaluate_tool_call(metadata, {"path": "demo.txt"})

    assert decision.allowed is False
    assert decision.requires_permission is False
    assert decision.blocked is True
    assert decision.reason == "Tool is not implemented: open_file."


@pytest.mark.parametrize(
    ("risk_level", "allowed", "requires_permission"),
    [
        (RiskLevel.read, True, False),
        (RiskLevel.low_write, True, False),
        (RiskLevel.medium_write, False, True),
        (RiskLevel.high_risk, False, True),
    ],
)
def test_policy_engine_applies_stage_one_risk_rules(
    risk_level: RiskLevel,
    allowed: bool,
    requires_permission: bool,
) -> None:
    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None
    metadata = metadata.model_copy(update={"risk_level": risk_level, "status": ToolStatus.implemented})

    decision = PolicyEngine().evaluate_tool_call(metadata, {"app_name": "notepad"})

    assert decision.allowed is allowed
    assert decision.requires_permission is requires_permission
    assert decision.blocked is False


def test_read_only_tools_execute_successfully() -> None:
    registry = build_default_registry()

    time_result = asyncio.run(registry.execute("get_current_time", {}))
    tools_result = asyncio.run(registry.execute("list_available_tools", {}))

    assert time_result.status == "success"
    assert "iso_time" in time_result.output
    assert tools_result.status == "success"
    assert any(tool["name"] == "launch_app" for tool in tools_result.output["tools"])
    assert any(tool["name"] == "get_current_time" for tool in tools_result.output["tools"])
    assert any(tool["name"] == "get_weather" for tool in tools_result.output["tools"])


def test_deterministic_llm_client_returns_valid_plan_shape() -> None:
    registry = build_default_registry()
    llm_client = DeterministicLLMClient()

    reply = asyncio.run(llm_client.complete([{"role": "user", "content": "hello"}]))
    plan = asyncio.run(llm_client.create_plan("hello", registry.list_tools()))

    assert reply == FALLBACK_MESSAGE
    assert isinstance(plan, AssistantPlan)
    assert plan.steps[0].title == "Produce normal assistant reply"


def test_openai_llm_client_complete_calls_model_with_vix_system_prompt() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Recursion is a function calling itself."))]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm_client = OpenAILLMClient(api_key="test-key", model="gpt-5.4-mini", client=fake_client)

    response = asyncio.run(llm_client.complete([{"role": "user", "content": "What is recursion?"}]))

    assert response == "Recursion is a function calling itself."
    assert completions.kwargs["model"] == "gpt-5.4-mini"
    assert completions.kwargs["messages"][0]["role"] == "system"
    assert "You are Vix" in completions.kwargs["messages"][0]["content"]
    assert completions.kwargs["messages"][1] == {"role": "user", "content": "What is recursion?"}


def test_openai_llm_client_generates_validated_proposed_tool_draft(caplog: pytest.LogCaptureFixture) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"name":"search_files","description":"Search local files.",'
                                '"reason":"No implemented file-search tool exists.",'
                                '"risk_level":"READ","input_schema":{"type":"object"},'
                                '"output_schema":{"type":"object"}}'
                            )
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm_client = OpenAILLMClient(api_key="test-key", model="gpt-5.4-mini", client=fake_client)

    caplog.set_level(logging.INFO, logger="app.assistant.llm_client")
    draft = asyncio.run(llm_client.propose_tool_spec("find my CV", build_default_registry().list_tools()))

    assert draft is not None
    assert draft.name == "search_files"
    assert draft.risk_level == "READ"
    assert completions.kwargs["model"] == "gpt-5.4-mini"
    assert completions.kwargs["response_format"]["type"] == "json_schema"
    assert completions.kwargs["response_format"]["json_schema"]["name"] == "proposed_tool_draft"
    assert "ai proposal raw output" in caplog.text
    assert "search_files" in caplog.text


def test_openai_llm_client_rejects_invalid_proposed_tool_risk_level() -> None:
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"name":"unsafe_tool","description":"Bad risk.",'
                                '"reason":"Bad risk.","risk_level":"MEDIUM",'
                                '"input_schema":{},"output_schema":{}}'
                            )
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm_client = OpenAILLMClient(api_key="test-key", client=fake_client)

    with pytest.raises(ValueError):
        asyncio.run(llm_client.propose_tool_spec("inspect screen", build_default_registry().list_tools()))


def test_ai_proposals_are_disabled_by_default() -> None:
    config = AppConfig(openai_api_key="test-key")

    assert config.ai_provider == "openai"
    assert config.ai_proposal_model == "gpt-5.4-mini"
    assert config.ai_general_answers_enabled is True
    assert config.ai_proposals_enabled is False
    assert isinstance(_build_llm_client(config), OpenAILLMClient)


def test_config_can_load_private_backend_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "AI_PROVIDER", "AI_PROPOSAL_MODEL", "AI_GENERAL_ANSWERS_ENABLED", "AI_PROPOSALS_ENABLED"):
        monkeypatch.delenv(name, raising=False)

    config_file = tmp_path / ".env"
    config_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "AI_PROVIDER=openai",
                "AI_PROPOSAL_MODEL=gpt-5.4-mini",
                "AI_GENERAL_ANSWERS_ENABLED=false",
                "AI_PROPOSALS_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config_from_file(config_file)

    assert config.config_file_path == config_file
    assert config.openai_api_key == "test-key"
    assert config.ai_provider == "openai"
    assert config.ai_proposal_model == "gpt-5.4-mini"
    assert config.ai_general_answers_enabled is False
    assert config.ai_proposals_enabled is True


def test_config_file_overrides_process_environment_for_ai_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-old-environment-key")
    monkeypatch.setenv("AI_PROPOSALS_ENABLED", "false")
    config_file = tmp_path / ".env"
    config_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-proj-new-file-key",
                "AI_PROVIDER=openai",
                "AI_PROPOSAL_MODEL=gpt-5.4-mini",
                "AI_PROPOSALS_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config_from_file(config_file)

    assert config.openai_api_key == "sk-proj-new-file-key"
    assert config.ai_proposals_enabled is True


def test_openai_client_is_selected_when_any_ai_capability_is_enabled() -> None:
    config = AppConfig(openai_api_key="test-key", ai_proposals_enabled=True)

    assert isinstance(_build_llm_client(config), OpenAILLMClient)


def test_ai_status_endpoint_reports_missing_key_when_general_answers_enabled(client: TestClient) -> None:
    response = client.get("/ai/status")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-5.4-mini"
    assert body["general_answers_enabled"] is True
    assert body["proposals_enabled"] is False
    assert body["api_key_configured"] is False
    assert body["model_reachable"] is False
    assert body["connected"] is False
    assert body["status"] == "missing_api_key"
    assert body["general_answers_status"] == "missing_api_key"
    assert body["tool_proposals_status"] == "disabled"
    assert body["api_key_status"] == "missing"
    assert body["model_status"] == "not_checked"


def test_ai_status_reports_disabled_when_all_ai_capabilities_are_disabled() -> None:
    config = AppConfig(openai_api_key="test-key", ai_general_answers_enabled=False, ai_proposals_enabled=False)

    status = asyncio.run(_get_ai_status(config, DeterministicLLMClient()))

    assert status.connected is False
    assert status.status == "disabled"
    assert status.general_answers_enabled is False
    assert status.proposals_enabled is False
    assert status.api_key_status == "configured"
    assert status.model_status == "not_checked"


def test_ai_status_reloads_private_config_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLMClient:
        async def verify_connection(self) -> tuple[bool, str]:
            return True, "fake OpenAI verification"

    config_file = tmp_path / ".env"
    config_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "AI_PROVIDER=openai",
                "AI_PROPOSAL_MODEL=gpt-5.4-mini",
                "AI_PROPOSALS_ENABLED=false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config_from_file(config_file)
    app = create_app(database_path=tmp_path / "reload.sqlite3", config=config, reload_config_from_file=True)

    def build_fake_llm_client(refreshed_config: AppConfig) -> FakeLLMClient:
        assert refreshed_config.openai_api_key == "test-key"
        assert refreshed_config.ai_proposals_enabled is True
        return FakeLLMClient()

    monkeypatch.setattr(main_module, "_build_llm_client", build_fake_llm_client)
    config_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "AI_PROVIDER=openai",
                "AI_PROPOSAL_MODEL=gpt-5.4-mini",
                "AI_PROPOSALS_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    with TestClient(app) as test_client:
        response = test_client.get("/ai/status")

    assert response.status_code == 200
    body = response.json()
    assert body["proposals_enabled"] is True
    assert body["api_key_configured"] is True
    assert body["model_reachable"] is True
    assert body["connected"] is True
    assert body["status"] == "connected"
    assert body["general_answers_status"] == "enabled"
    assert body["tool_proposals_status"] == "enabled"
    assert body["api_key_status"] == "configured"
    assert body["model_status"] == "reachable"
    assert body["detail"] == "fake OpenAI verification"


def test_ai_status_reports_missing_api_key_when_enabled() -> None:
    config = AppConfig(openai_api_key=None, ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, DeterministicLLMClient()))

    assert status.connected is False
    assert status.status == "missing_api_key"
    assert status.general_answers_status == "missing_api_key"
    assert status.tool_proposals_status == "missing_api_key"
    assert status.api_key_status == "missing"
    assert status.model_status == "not_checked"
    assert status.detail == "OPENAI_API_KEY is not configured in backend/.env."


def test_ai_status_verifies_configured_model() -> None:
    class FakeLLMClient(DeterministicLLMClient):
        async def verify_connection(self) -> tuple[bool, str]:
            return True, "verified"

    config = AppConfig(openai_api_key="test-key", ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, FakeLLMClient()))

    assert status.connected is True
    assert status.model_reachable is True
    assert status.status == "connected"
    assert status.general_answers_status == "enabled"
    assert status.tool_proposals_status == "enabled"
    assert status.model_status == "reachable"
    assert status.detail == "verified"


def test_ai_status_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowLLMClient(DeterministicLLMClient):
        async def verify_connection(self) -> tuple[bool, str]:
            return True, "too late"

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(main_module.asyncio, "wait_for", fake_wait_for)
    config = AppConfig(openai_api_key="test-key", ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, SlowLLMClient()))

    assert status.connected is False
    assert status.model_reachable is False
    assert status.status == "verification_failed"
    assert status.general_answers_status == "model_unreachable"
    assert status.tool_proposals_status == "model_unreachable"
    assert status.model_status == "unreachable"
    assert status.detail == "OpenAI model verification timed out after 8 seconds."


def test_ai_status_redacts_api_key_from_openai_errors() -> None:
    class InvalidKeyLLMClient(DeterministicLLMClient):
        async def verify_connection(self) -> tuple[bool, str]:
            return False, "OpenAI model verification failed: Incorrect API key provided: sk-proj-secret123456"

    config = AppConfig(openai_api_key="sk-proj-secret123456", ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, InvalidKeyLLMClient()))

    assert status.connected is False
    assert "sk-proj-secret123456" not in status.detail


def test_normal_chat_uses_llm_reply_without_tool_execution(tmp_path) -> None:
    class FakeLLMClient:
        def __init__(self) -> None:
            self.complete_messages: list[list[dict[str, object]]] = []

        async def complete(self, messages: list[dict[str, object]]) -> str:
            self.complete_messages.append(messages)
            return "Normal assistant reply."

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            return None

    llm_client = FakeLLMClient()
    app = create_app(database_path=tmp_path / "llm.sqlite3", llm_client=llm_client, config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "What is recursion?"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(7)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "llm_response_started",
        "llm_response_finished",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "GENERAL_ANSWER"
    assert events[-1]["data"]["message"] == "Normal assistant reply."
    assert llm_client.complete_messages[0][-1] == {"role": "user", "content": "What is recursion?"}
    assert app.state.database.count_rows("tool_calls") == 0


def test_session_history_is_sent_to_llm_for_follow_up(tmp_path) -> None:
    class FakeLLMClient:
        def __init__(self) -> None:
            self.complete_messages: list[list[dict[str, object]]] = []

        async def complete(self, messages: list[dict[str, object]]) -> str:
            self.complete_messages.append(messages)
            return "Example reply."

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            return None

    llm_client = FakeLLMClient()
    app = create_app(database_path=tmp_path / "memory.sqlite3", llm_client=llm_client, config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            first = test_client.post("/chat", json={"message": "What is recursion?"})
            assert first.status_code == 200
            first_run = [websocket.receive_json() for _ in range(7)]
            conversation_id = first.json()["conversation_id"]

            second = test_client.post(
                "/chat",
                json={"message": "give me Python example", "conversation_id": conversation_id},
            )
            assert second.status_code == 200
            second_run = [websocket.receive_json() for _ in range(7)]

    assert first_run[-1]["type"] == "assistant_message_created"
    assert second_run[-1]["type"] == "assistant_message_created"
    second_messages = llm_client.complete_messages[1]
    assert {"role": "user", "content": "What is recursion?"} in second_messages
    assert {"role": "assistant", "content": "Example reply."} in second_messages
    assert second_messages[-1] == {"role": "user", "content": "give me Python example"}


def test_permission_required_event_contains_structured_preview(tmp_path) -> None:
    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None
    metadata = metadata.model_copy(update={"confirmation_policy": ConfirmationPolicy.before_execute})

    async def fake_launch(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool="launch_app",
            status="success",
            output={"status": "success", "message": f"Test launch for {arguments['app_name']}."},
        )

    registry.register(metadata, fake_launch)
    app = create_app(database_path=tmp_path / "permissions.sqlite3", registry=registry, config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "open notepad"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(6)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "tool_selected",
        "permission_required",
    ]
    permission_event = events[-1]
    preview = permission_event["data"]["preview"]
    assert preview["permission_id"] == permission_event["data"]["permission_id"]
    assert preview["action"] == "Run launch_app"
    assert preview["target"] == "notepad"
    assert preview["content"] == {"app_name": "notepad"}
    assert preview["risk_level"] == "LOW_WRITE"
    assert preview["what_will_happen"] == "The assistant will launch the whitelisted Windows application 'notepad'."
    assert preview["editable"] is False
    assert preview["edit_schema"]["properties"]["app_name"]["type"] == "string"
    assert app.state.database.count_rows("permissions") == 1


@pytest.mark.parametrize(
    "message",
    [
        "open calculator",
        "please run calculator",
        "yes, open calculator",
        "please open the thing used for calculating things, named calculator",
    ],
)
def test_calculator_launch_requests_use_local_tool_not_llm(tmp_path, message: str) -> None:
    class FailingLLMClient:
        async def complete(self, messages: list[dict[str, object]]) -> str:
            raise AssertionError("LLM complete should not be called for local tools.")

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            raise AssertionError("Tool proposal should not be called for local tools.")

    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None

    async def fake_launch(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool="launch_app",
            status="success",
            output={"status": "success", "message": f"Test launch for {arguments['app_name']}."},
        )

    registry.register(metadata, fake_launch)
    app = create_app(
        database_path=tmp_path / "calculator.sqlite3",
        registry=registry,
        llm_client=FailingLLMClient(),
        config=AppConfig(),
    )
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": message})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(8)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "tool_selected",
        "tool_started",
        "tool_result",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "LOCAL_TOOL"
    assert events[4]["data"]["tool_name"] == "launch_app"
    assert app.state.database.count_rows("tool_calls") == 1


def test_weather_today_without_location_asks_for_city(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "what is weather today"})
        assert response.status_code == 200

        events = [websocket.receive_json() for _ in range(5)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "REALTIME_INFO"
    assert events[2]["data"]["missing_input"] == "location"
    assert events[-1]["data"]["message"] == "Which city should I check?"
    assert client.app.state.database.count_rows("tool_calls") == 0


@pytest.mark.parametrize(
    ("message", "location", "requested_date"),
    [
        ("what's the weather in New York tomorrow?", "New York", "tomorrow"),
        ("weather in Bielsko-Biała", "Bielsko-Biała", "today"),
        ("weather in São Paulo", "São Paulo", "today"),
        ("weather at 7pm in Warsaw", "Warsaw", "today"),
        ("will it rain in Warsaw this weekend?", "Warsaw", "today"),
    ],
)
def test_weather_classifier_extracts_multiword_and_unicode_locations(
    message: str,
    location: str,
    requested_date: str,
) -> None:
    classification = classify_request(message)

    assert classification.category == RequestCategory.realtime_info
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_weather"
    assert classification.tool_proposal.arguments == {"location": location, "date": requested_date}


@pytest.mark.parametrize("message", ["weather at 7pm", "will it rain this weekend?"])
def test_weather_classifier_asks_for_location_when_ambiguous(message: str) -> None:
    classification = classify_request(message)

    assert classification.category == RequestCategory.realtime_info
    assert classification.tool_proposal is None
    assert classification.missing_input == "location"


def test_vix_window_is_ignored_by_tracker() -> None:
    tracker = WindowTracker()
    vscode = _window(hwnd=101, title="main.py - Visual Studio Code", process_name="Code.exe")
    vix = _window(hwnd=202, title="Desktop Assistant", process_name="DesktopAssistant.Frontend.exe", is_vix=True)

    tracker.on_foreground_window_changed(vscode)
    tracker.on_foreground_window_changed(vix)

    assert tracker.get_current_foreground_window(refresh=False) == vix
    assert tracker.get_context_window(validate_exists=False) == vscode
    assert tracker.snapshot().last_non_vix_window == vscode


def test_last_non_vix_window_is_preserved_when_vix_gets_focus() -> None:
    tracker = WindowTracker()
    chrome = _window(hwnd=303, title="Article - Google Chrome", process_name="chrome.exe")
    vix = _window(hwnd=404, title="Desktop Assistant", process_name="DesktopAssistant.Frontend.exe", is_vix=True)

    tracker.on_foreground_window_changed(chrome)
    tracker.on_foreground_window_changed(vix)

    assert tracker.get_context_window(validate_exists=False).title == "Article - Google Chrome"
    assert tracker.get_current_foreground_window(refresh=False).is_vix is True


def test_context_window_fallback_does_not_overwrite_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = WindowTracker()
    vix = _window(hwnd=405, title="Desktop Assistant", process_name="DesktopAssistant.Frontend.exe", is_vix=True)
    chrome = _window(hwnd=406, title="asd - Google Search - Google Chrome", process_name="chrome.exe")
    tracker.on_foreground_window_changed(vix)
    monkeypatch.setattr(tracker, "_find_fallback_context_window", lambda: chrome)

    context = tracker.get_context_window(validate_exists=False)

    assert context == chrome
    assert tracker.get_current_foreground_window(refresh=False) == vix


def test_get_context_window_returns_last_non_vix_window() -> None:
    vscode = _window(hwnd=505, title="main.py - Visual Studio Code", process_name="Code.exe")
    executors = build_context_tool_executors(FakeWindowTracker(context=vscode))

    result = asyncio.run(executors["get_context_window_info"]({}))

    assert result.status == "success"
    assert result.output["window"]["title"] == "main.py - Visual Studio Code"
    assert result.output["source"] == "context_window"


def test_get_foreground_window_can_return_vix() -> None:
    vix = _window(hwnd=606, title="Desktop Assistant", process_name="DesktopAssistant.Frontend.exe", is_vix=True)
    executors = build_context_tool_executors(FakeWindowTracker(current=vix))

    result = asyncio.run(executors["get_foreground_window_info"]({}))

    assert result.status == "success"
    assert result.output["window"]["is_vix"] is True
    assert result.output["source"] == "foreground_window"


def test_get_selected_text_fails_if_no_context_window() -> None:
    executors = build_context_tool_executors(FakeWindowTracker())

    result = asyncio.run(executors["get_selected_text"]({}))

    assert result.status == "failed"
    assert result.error == "No previous non-Vix window was captured."


def test_get_clipboard_text_handles_empty_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_tools,
        "_try_read_clipboard_text",
        lambda: context_tools._ClipboardReadResult(False, error="Clipboard does not contain text."),
    )
    executors = build_context_tool_executors(FakeWindowTracker())

    result = asyncio.run(executors["get_clipboard_text"]({}))

    assert result.status == "failed"
    assert result.error == "Clipboard does not contain text."


def test_get_clipboard_text_handles_access_violation_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_access_violation():
        raise OSError("exception: access violation reading 0xFFFFFFFF9D35EC10")

    monkeypatch.setattr(context_tools, "_try_read_clipboard_text", raise_access_violation)
    executors = build_context_tool_executors(FakeWindowTracker())

    result = asyncio.run(executors["get_clipboard_text"]({}))

    assert result.status == "failed"
    assert "access violation" in result.error


def test_classifier_selected_text_routes_to_local_context() -> None:
    classification = classify_request("What does selected word mean?")

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_selected_text"


def test_classifier_selected_in_google_routes_to_local_context() -> None:
    classification = classify_request("what text did i select in google?")

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_selected_text"


def test_classifier_selected_in_foreground_window_routes_to_selected_text() -> None:
    classification = classify_request("what text do i have selected in foreground window?")

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_selected_text"


def test_classifier_selected_ungrammatical_phrase_routes_to_selected_text() -> None:
    classification = classify_request("what text i have selected")

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_selected_text"


def test_classifier_current_window_uses_context_window_not_foreground_window() -> None:
    classification = classify_request("What app was I using?")

    assert classification.category == RequestCategory.local_context
    assert classification.tool_proposal is not None
    assert classification.tool_proposal.name == "get_context_window_info"


def test_get_weather_output_includes_scientific_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    today = date.today().isoformat()

    def fake_get_json(url: str, params: dict[str, object]) -> tuple[dict[str, object], str]:
        if url == weather_tools.GEOCODING_URL:
            return (
                {
                    "generationtime_ms": 0.23,
                    "results": [
                        {
                            "name": "Warsaw",
                            "admin1": "Masovian Voivodeship",
                            "country": "Poland",
                            "latitude": 52.23,
                            "longitude": 21.01,
                        },
                        {
                            "name": "Warsaw",
                            "admin1": "Indiana",
                            "country": "United States",
                            "latitude": 41.24,
                            "longitude": -85.85,
                        },
                    ],
                },
                "https://geocoding-api.open-meteo.com/v1/search?name=Warsaw&count=5",
            )

        assert url == weather_tools.FORECAST_URL
        return (
            {
                "generationtime_ms": 0.78,
                "timezone": "Europe/Warsaw",
                "current": {"temperature_2m": 20.5, "wind_speed_10m": 8.0},
                "daily": {
                    "time": [today],
                    "weather_code": [0],
                    "temperature_2m_max": [24.0],
                    "temperature_2m_min": [15.0],
                    "precipitation_probability_max": [10],
                    "wind_speed_10m_max": [18.0],
                },
            },
            "https://api.open-meteo.com/v1/forecast?latitude=52.23&longitude=21.01",
        )

    monkeypatch.setattr(weather_tools, "_get_json", fake_get_json)

    result = weather_tools._get_weather_sync("Warsaw", "today")

    assert result.status == "success"
    output = result.output
    assert output["source"] == "Open-Meteo"
    assert output["source_urls"]["geocoding"].startswith("https://geocoding-api.open-meteo.com")
    assert output["source_urls"]["forecast"].startswith("https://api.open-meteo.com")
    assert output["resolved_coordinates"] == {"latitude": 52.23, "longitude": 21.01}
    assert output["timezone"] == "Europe/Warsaw"
    assert output["units"] == {
        "temperature": "celsius",
        "wind_speed": "km/h",
        "precipitation_probability": "percent",
    }
    assert output["forecast_generation_time_ms"] == 0.78
    assert output["geocoding"]["candidate_count"] == 2
    assert output["geocoding"]["is_ambiguous"] is True
    assert output["geocoding"]["selected_name"] == "Warsaw, Masovian Voivodeship, Poland"


def test_weather_today_in_warsaw_calls_get_weather(tmp_path) -> None:
    registry = build_default_registry()
    metadata = registry.get("get_weather")
    assert metadata is not None

    async def fake_weather(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool="get_weather",
            status="success",
            output={
                "status": "success",
                "message": f"Weather for {arguments['location']}: Clear sky, 20 C.",
                "location": arguments["location"],
                "temperature": {"current_c": 20},
                "condition": "Clear sky",
                "precipitation_probability": 0,
                "wind": {"current_kmh": 8},
                "source": "Fake weather",
            },
        )

    registry.register(metadata, fake_weather)
    app = create_app(database_path=tmp_path / "weather.sqlite3", registry=registry, config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "what is weather today in Warsaw"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(8)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "tool_selected",
        "tool_started",
        "tool_result",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "REALTIME_INFO"
    assert events[4]["data"]["tool_name"] == "get_weather"
    assert events[4]["data"]["arguments"] == {"location": "Warsaw", "date": "today"}
    assert events[-1]["data"]["message"] == "Weather for Warsaw: Clear sky, 20 C."
    assert app.state.database.count_rows("tool_calls") == 1


def test_llm_failure_returns_graceful_assistant_error(tmp_path) -> None:
    class FailingLLMClient:
        async def complete(self, messages: list[dict[str, object]]) -> str:
            raise RuntimeError("model unavailable")

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            return None

    app = create_app(database_path=tmp_path / "llm-failure.sqlite3", llm_client=FailingLLMClient(), config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "What is recursion?"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(7)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "llm_response_started",
        "llm_response_finished",
        "assistant_message_created",
    ]
    assert events[5]["data"]["status"] == "failed"
    assert events[-1]["data"]["message"] == "I could not get an AI answer right now. Please try again."


def test_unsafe_request_is_blocked(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "delete system32"})
        assert response.status_code == 200

        events = [websocket.receive_json() for _ in range(6)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "error_occurred",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "UNSAFE_OR_BLOCKED"
    assert events[-1]["data"]["message"] == "Request appears to ask for unsafe or destructive behavior."
    assert client.app.state.database.count_rows("tool_calls") == 0


def test_unknown_chat_request_without_missing_capability_uses_llm_fallback(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "tell me a joke about databases"})
        assert response.status_code == 200

        event_types = []
        assistant_message = None
        for _ in range(7):
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "assistant_message_created":
                assistant_message = event["data"]["message"]

        assert event_types == [
            "user_message_received",
            "assistant_thinking_started",
            "request_classified",
            "plan_created",
            "llm_response_started",
            "llm_response_finished",
            "assistant_message_created",
        ]
        assert assistant_message == FALLBACK_MESSAGE


def test_unknown_open_app_creates_proposed_tool(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "open spotify"})
        assert response.status_code == 200

        events = [websocket.receive_json() for _ in range(6)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "proposed_tool_created",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "MISSING_TOOL"
    assert events[4]["data"]["name"] == "launch_additional_app"
    assert events[-1]["data"]["message"] == "I cannot do this yet, but I proposed a new tool: launch_additional_app."


def test_llm_refusal_for_action_request_creates_proposed_tool(tmp_path) -> None:
    class RefusingLLMClient:
        async def complete(self, messages: list[dict[str, object]]) -> str:
            return "I can't directly manage your audio mixer yet. You can open Windows sound settings manually."

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            return ProposedToolDraft(
                name="control_audio_mixer",
                description="Adjust local audio mixer levels after user review.",
                reason="The user asked Vix to manage desktop audio mixer levels, but no audio-control tool exists.",
                risk_level="LOW_WRITE",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )

    app = create_app(database_path=tmp_path / "post-llm-proposal.sqlite3", llm_client=RefusingLLMClient(), config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "please manage my audio mixer levels"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(8)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "llm_response_started",
        "llm_response_finished",
        "proposed_tool_created",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "GENERAL_ANSWER"
    assert events[6]["data"]["name"] == "control_audio_mixer"
    assert "I also proposed a new tool for developer review: control_audio_mixer." in events[-1]["data"]["message"]
    assert app.state.database.count_rows("proposed_tools") == 1


def test_unsupported_file_request_creates_proposed_tool(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "find my kafka presentation"})
        assert response.status_code == 200

        events = [websocket.receive_json() for _ in range(6)]

    event_types = [event["type"] for event in events]
    assert event_types == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "proposed_tool_created",
        "assistant_message_created",
    ]
    proposed_event = events[4]
    assert proposed_event["data"]["name"] == "search_files"
    assert proposed_event["data"]["risk_level"] == "READ"
    assert events[-1]["data"]["message"] == "I cannot do this yet, but I proposed a new tool: search_files."

    response = client.get("/proposed-tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    assert [tool["name"] for tool in tools] == ["search_files"]
    assert tools[0]["status"] == "proposed"
    assert tools[0]["created_from_message"] == "find my kafka presentation"
    assert client.app.state.database.count_rows("proposed_tools") == 1
    assert client.app.state.database.count_rows("reflections") == 1


def test_unknown_screen_request_creates_capture_window_proposal(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "look at my screen and explain this"})
        assert response.status_code == 200

        events = [websocket.receive_json() for _ in range(6)]

    assert events[2]["type"] == "request_classified"
    assert events[2]["data"]["category"] == "MISSING_TOOL"
    assert events[4]["type"] == "proposed_tool_created"
    assert events[4]["data"]["name"] == "capture_active_window"
    assert events[-1]["data"]["message"] == "I cannot do this yet, but I proposed a new tool: capture_active_window."


def test_list_proposed_tools_returns_created_tool(client: TestClient) -> None:
    payload = {
        "name": "search_files",
        "description": "Search local files.",
        "reason": "User asked for file search.",
        "risk_level": "READ",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "created_from_message": "find my CV",
    }

    create_response = client.post("/proposed-tools", json=payload)
    list_response = client.get("/proposed-tools")

    assert create_response.status_code == 200
    assert list_response.status_code == 200
    tools = list_response.json()["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "search_files"
    assert tools[0]["input_schema"] == {"type": "object"}


def test_proposed_tool_created_event_is_emitted(client: TestClient) -> None:
    payload = {
        "name": "search_files",
        "description": "Search local files.",
        "reason": "User asked for file search.",
        "risk_level": "READ",
        "created_from_message": "find my CV",
    }

    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/proposed-tools", json=payload)
        assert response.status_code == 200

        event = websocket.receive_json()

    assert event["type"] == "proposed_tool_created"
    assert event["data"]["name"] == "search_files"
    assert event["data"]["tool"]["status"] == "proposed"


def test_approve_proposed_tool_changes_status(client: TestClient) -> None:
    create_response = client.post(
        "/proposed-tools",
        json={
            "name": "search_files",
            "description": "Search local files.",
            "reason": "User asked for file search.",
            "risk_level": "READ",
            "created_from_message": "find my CV",
        },
    )
    tool_id = create_response.json()["id"]

    response = client.post(f"/proposed-tools/{tool_id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert client.get("/proposed-tools").json()["tools"][0]["status"] == "approved"


def test_reject_proposed_tool_changes_status(client: TestClient) -> None:
    create_response = client.post(
        "/proposed-tools",
        json={
            "name": "search_files",
            "description": "Search local files.",
            "reason": "User asked for file search.",
            "risk_level": "READ",
            "created_from_message": "find my CV",
        },
    )
    tool_id = create_response.json()["id"]

    response = client.post(f"/proposed-tools/{tool_id}/reject")

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert client.get("/proposed-tools").json()["tools"][0]["status"] == "rejected"


def test_needs_changes_proposed_tool_changes_status(client: TestClient) -> None:
    create_response = client.post(
        "/proposed-tools",
        json={
            "name": "search_files",
            "description": "Search local files.",
            "reason": "User asked for file search.",
            "risk_level": "READ",
            "created_from_message": "find my CV",
        },
    )
    tool_id = create_response.json()["id"]

    response = client.post(f"/proposed-tools/{tool_id}/needs-changes")

    assert response.status_code == 200
    assert response.json()["status"] == "needs_changes"


def test_llm_tool_proposal_is_persisted_without_execution(tmp_path) -> None:
    class FakeLLMClient:
        async def complete(self, messages: list[dict[str, object]]) -> str:
            return FALLBACK_MESSAGE

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            return ProposedToolDraft(
                name="summarize_documents",
                description="Summarize selected local documents.",
                reason="The user asked for document summarization and no implemented tool exists.",
                risk_level="READ",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )

    app = create_app(database_path=tmp_path / "proposal.sqlite3", llm_client=FakeLLMClient(), config=AppConfig())
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"
            response = test_client.post("/chat", json={"message": "summarize my invoices"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(6)]

    assert events[4]["type"] == "proposed_tool_created"
    assert events[4]["data"]["name"] == "summarize_documents"
    assert app.state.database.count_rows("tool_calls") == 0
    assert app.state.database.count_rows("proposed_tools") == 1


def test_selected_text_request_captures_artifact_and_uses_llm_context(tmp_path) -> None:
    class CapturingLLMClient:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def complete(self, messages: list[dict[str, object]]) -> str:
            self.messages = messages
            return "Recursion is when a definition or function refers back to itself."

        async def create_plan(self, user_text: str, tools: list[object]) -> AssistantPlan:
            return AssistantPlan(goal="Fake plan", steps=[])

        async def propose_tool_spec(self, user_message: str, existing_tools: list[object]) -> ProposedToolDraft | None:
            raise AssertionError("Local context requests should not create proposed tools.")

    context_window = _window(hwnd=707, title="notes.txt - Notepad", process_name="notepad.exe")
    tracker = FakeWindowTracker(context=context_window)
    registry = build_default_registry(tracker, FakeSelectedTextStrategy())
    llm_client = CapturingLLMClient()
    app = create_app(
        database_path=tmp_path / "selected-text.sqlite3",
        registry=registry,
        llm_client=llm_client,
        config=AppConfig(),
        context_tracker=tracker,
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = test_client.post("/chat", json={"message": "What does selected word mean?"})
            assert response.status_code == 200

            events = [websocket.receive_json() for _ in range(13)]

        context_response = test_client.get("/context/status")

    event_types = [event["type"] for event in events]
    assert event_types == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "context_window_selected",
        "tool_selected",
        "tool_started",
        "tool_result",
        "context_captured",
        "artifact_created",
        "llm_response_started",
        "llm_response_finished",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "LOCAL_CONTEXT"
    assert events[5]["data"]["tool_name"] == "get_selected_text"
    assert events[8]["data"]["artifact_type"] == "selected_text"
    assert events[8]["data"]["content_preview"] == "recursion"
    assert events[9]["data"]["artifact"]["type"] == "selected_text"
    assert events[9]["data"]["artifact"]["content_text"] == "recursion"
    assert "recursion" in llm_client.messages[-1]["content"]
    assert context_response.status_code == 200
    assert context_response.json()["last_context_artifact"]["content_text"] == "recursion"
    assert app.state.database.count_rows("artifacts") == 1
    assert app.state.database.count_rows("tool_calls") == 1


def test_context_status_endpoint_returns_tracker_snapshot(tmp_path) -> None:
    context_window = _window(hwnd=808, title="main.py - Visual Studio Code", process_name="Code.exe")
    foreground = _window(hwnd=909, title="Desktop Assistant", process_name="DesktopAssistant.Frontend.exe", is_vix=True)
    tracker = FakeWindowTracker(current=foreground, context=context_window)
    app = create_app(database_path=tmp_path / "context-status.sqlite3", config=AppConfig(), context_tracker=tracker)

    with TestClient(app) as test_client:
        response = test_client.get("/context/status")

    assert response.status_code == 200
    body = response.json()
    assert body["current_foreground_window"]["is_vix"] is True
    assert body["last_context_window"]["process_name"] == "Code.exe"
    assert body["last_context_artifact"] is None


def test_chat_persists_trace_tables_from_run_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        response = client.post("/chat", json={"message": "open notepad"})
        assert response.status_code == 200

        event_types = [websocket.receive_json()["type"] for _ in range(8)]
        assert event_types == [
            "user_message_received",
            "assistant_thinking_started",
            "request_classified",
            "plan_created",
            "tool_selected",
            "tool_started",
            "tool_result",
            "assistant_message_created",
        ]

    database = client.app.state.database
    assert database.count_rows("sessions") == 1
    assert database.count_rows("messages") == 2
    assert database.count_rows("events") >= 7
    assert database.count_rows("tool_calls") == 1


def test_permission_decisions_emit_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        assert websocket.receive_json()["type"] == "event_stream_connected"

        approve = client.post("/permissions/demo-1/approve")
        reject = client.post("/permissions/demo-2/reject")

        assert approve.status_code == 200
        assert approve.json() == {"permission_id": "demo-1", "status": "approved"}
        assert reject.status_code == 200
        assert reject.json() == {"permission_id": "demo-2", "status": "rejected"}
        assert websocket.receive_json()["type"] == "permission_approved"
        assert websocket.receive_json()["type"] == "permission_rejected"
