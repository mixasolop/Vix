import asyncio
from collections.abc import Iterator
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.assistant.llm_client import DeterministicLLMClient, OpenAILLMClient
from app.assistant.planner import FALLBACK_MESSAGE, Planner
from app.assistant.policy import PolicyEngine
from app.config import AppConfig, load_config_from_file
from app.main import _build_llm_client, _get_ai_status, create_app
from app.schemas.plans import AssistantPlan
from app.schemas.proposed_tools import ProposedToolDraft
from app.schemas.tools import ConfirmationPolicy, RiskLevel, ToolResult, ToolStatus
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


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["stage"] == "3.0"


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
    assert body["connected"] is False
    assert body["status"] == "missing_api_key"


def test_ai_status_reports_disabled_when_all_ai_capabilities_are_disabled() -> None:
    config = AppConfig(openai_api_key="test-key", ai_general_answers_enabled=False, ai_proposals_enabled=False)

    status = asyncio.run(_get_ai_status(config, DeterministicLLMClient()))

    assert status.connected is False
    assert status.status == "disabled"
    assert status.general_answers_enabled is False
    assert status.proposals_enabled is False


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
    assert body["connected"] is True
    assert body["status"] == "connected"
    assert body["detail"] == "fake OpenAI verification"


def test_ai_status_reports_missing_api_key_when_enabled() -> None:
    config = AppConfig(openai_api_key=None, ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, DeterministicLLMClient()))

    assert status.connected is False
    assert status.status == "missing_api_key"
    assert status.detail == "OPENAI_API_KEY is not configured in backend/.env."


def test_ai_status_verifies_configured_model() -> None:
    class FakeLLMClient(DeterministicLLMClient):
        async def verify_connection(self) -> tuple[bool, str]:
            return True, "verified"

    config = AppConfig(openai_api_key="test-key", ai_proposals_enabled=True)

    status = asyncio.run(_get_ai_status(config, FakeLLMClient()))

    assert status.connected is True
    assert status.status == "connected"
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
    assert status.status == "verification_failed"
    assert status.detail == "OpenAI model verification timed out after 8 seconds."


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


def test_open_calculator_uses_local_tool_not_llm(tmp_path) -> None:
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

            response = test_client.post("/chat", json={"message": "open calculator"})
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
