from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.main import create_app
from app.schemas.tools import ToolResult
from app.tools.registry import build_default_registry


def test_bare_calc_requests_clarification_without_tool_call(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = client.post("/chat", json={"message": "calc"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(6)]

    assert [event["type"] for event in events] == [
        "user_message_received",
        "assistant_thinking_started",
        "request_classified",
        "plan_created",
        "clarification_required",
        "assistant_message_created",
    ]
    assert events[2]["data"]["category"] == "ASK_CLARIFICATION"
    assert events[4]["data"]["proposed_tool_name"] == "launch_app"
    assert events[4]["data"]["proposed_arguments"] == {"app_name": "calculator"}
    assert events[-1]["data"]["message"] == "Do you want me to open Calculator?"
    assert launched_apps == []
    assert app.state.database.count_rows("clarifications") == 1
    assert app.state.database.count_rows("tool_calls") == 0


def test_bare_calculator_requests_clarification_without_tool_call(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = client.post("/chat", json={"message": "calculator"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(6)]

    assert events[2]["data"]["category"] == "ASK_CLARIFICATION"
    assert events[4]["type"] == "clarification_required"
    assert events[4]["data"]["proposed_arguments"] == {"app_name": "calculator"}
    assert launched_apps == []
    assert app.state.database.count_rows("tool_calls") == 0


def test_clarification_accept_executes_stored_tool_in_same_session(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            first = client.post("/chat", json={"message": "calc"})
            assert first.status_code == 200
            _ = [websocket.receive_json() for _ in range(6)]
            conversation_id = first.json()["conversation_id"]

            second = client.post("/chat", json={"message": "yes", "conversation_id": conversation_id})
            assert second.status_code == 200
            accepted_events = [websocket.receive_json() for _ in range(8)]

    assert [event["type"] for event in accepted_events] == [
        "user_message_received",
        "assistant_thinking_started",
        "clarification_accepted",
        "plan_created",
        "tool_selected",
        "tool_started",
        "tool_result",
        "assistant_message_created",
    ]
    assert accepted_events[2]["data"]["proposed_tool_name"] == "launch_app"
    assert accepted_events[4]["data"]["arguments"] == {"app_name": "calculator"}
    assert launched_apps == ["calculator"]
    assert app.state.database.count_rows("tool_calls") == 1


def test_clarification_reject_takes_no_action(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            first = client.post("/chat", json={"message": "calc"})
            assert first.status_code == 200
            _ = [websocket.receive_json() for _ in range(6)]
            conversation_id = first.json()["conversation_id"]

            second = client.post("/chat", json={"message": "no", "conversation_id": conversation_id})
            assert second.status_code == 200
            rejected_events = [websocket.receive_json() for _ in range(4)]

    assert [event["type"] for event in rejected_events] == [
        "user_message_received",
        "assistant_thinking_started",
        "clarification_rejected",
        "assistant_message_created",
    ]
    assert rejected_events[-1]["data"]["message"] == "Okay, no action taken."
    assert launched_apps == []
    assert app.state.database.count_rows("tool_calls") == 0


def test_yes_without_pending_clarification_does_not_execute_tool(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            response = client.post("/chat", json={"message": "yes"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(7)]

    assert events[2]["data"]["category"] == "GENERAL_ANSWER"
    assert "tool_selected" not in {event["type"] for event in events}
    assert launched_apps == []
    assert app.state.database.count_rows("tool_calls") == 0


def test_clarification_is_session_scoped(tmp_path) -> None:
    app, launched_apps = _clarification_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"

            first = client.post("/chat", json={"message": "calc", "conversation_id": "session-a"})
            assert first.status_code == 200
            _ = [websocket.receive_json() for _ in range(6)]

            second = client.post("/chat", json={"message": "yes", "conversation_id": "session-b"})
            assert second.status_code == 200
            session_b_events = [websocket.receive_json() for _ in range(7)]

    assert session_b_events[2]["data"]["category"] == "GENERAL_ANSWER"
    assert "tool_selected" not in {event["type"] for event in session_b_events}
    assert launched_apps == []
    assert app.state.database.count_rows("tool_calls") == 0


def _clarification_app(tmp_path) -> tuple[object, list[str]]:
    registry = build_default_registry()
    metadata = registry.get("launch_app")
    assert metadata is not None
    launched_apps: list[str] = []

    async def fake_launch(arguments: dict[str, object]) -> ToolResult:
        launched_apps.append(str(arguments["app_name"]))
        return ToolResult(
            tool="launch_app",
            status="success",
            output={"status": "success", "message": f"Test launch for {arguments['app_name']}."},
        )

    registry.register(metadata, fake_launch)
    app = create_app(database_path=tmp_path / "clarifications.sqlite3", registry=registry, config=AppConfig())
    return app, launched_apps
