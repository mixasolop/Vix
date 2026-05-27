from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path) -> Iterator[TestClient]:
    app = create_app(database_path=tmp_path / "events.sqlite3")
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tools_endpoint_lists_stage_one_contracts(client: TestClient) -> None:
    response = client.get("/tools")

    assert response.status_code == 200
    tool_names = {tool["name"] for tool in response.json()["tools"]}
    assert "launch_app" in tool_names
    assert "transcribe_audio" in tool_names


def test_chat_returns_fake_plan_and_tool_trace(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "open notepad"})

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["goal"] == "Open Notepad"
    assert body["tool_calls"][0]["tool"] == "launch_app"
    assert body["tool_calls"][0]["status"] == "success"


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

        websocket.send_json({"type": "client.ping"})
        echo_event = websocket.receive_json()
        assert echo_event["type"] == "client.event.received"
        assert echo_event["payload"]["received"] == {"type": "client.ping"}
