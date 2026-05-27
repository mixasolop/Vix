from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.tools import ToolResult
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
    app = create_app(database_path=tmp_path / "events.sqlite3", registry=registry)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tools_endpoint_lists_stage_one_contracts(client: TestClient) -> None:
    response = client.get("/tools")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()["tools"]}
    assert tools["launch_app"]["status"] == "implemented"
    assert tools["transcribe_audio"]["status"] == "planned"


def test_chat_returns_fake_plan_and_tool_trace(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "open notepad"})

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["goal"] == "Open Notepad"
    assert body["tool_calls"][0]["tool"] == "launch_app"
    assert body["tool_calls"][0]["status"] == "success"
    assert body["assistant_message"] == "Test launch for notepad."


def test_chat_persists_trace_tables(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "open notepad"})

    assert response.status_code == 200
    database = client.app.state.database
    assert database.count_rows("sessions") == 1
    assert database.count_rows("messages") == 2
    assert database.count_rows("events") >= 5
    assert database.count_rows("tool_calls") == 1


def test_permission_decisions(client: TestClient) -> None:
    approve = client.post("/permissions/demo-1/approve")
    reject = client.post("/permissions/demo-2/reject")

    assert approve.status_code == 200
    assert approve.json() == {"permission_id": "demo-1", "status": "approved"}
    assert reject.status_code == 200
    assert reject.json() == {"permission_id": "demo-2", "status": "rejected"}


def test_websocket_event_stream(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as websocket:
        ready_event = websocket.receive_json()
        assert ready_event["type"] == "event_stream.connected"

        response = client.post("/chat", json={"message": "open notepad"})
        assert response.status_code == 200

        event_types = [websocket.receive_json()["type"] for _ in range(5)]
        assert event_types == [
            "chat.message.received",
            "plan.created",
            "tool.started",
            "tool.finished",
            "assistant.message.created",
        ]
