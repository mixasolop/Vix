from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.browser.session_manager import BrowserSessionManager
from app.config import AppConfig
from app.main import create_app
from app.tools.registry import build_default_registry


PAGES_DIR = Path(__file__).parent / "fixtures" / "pages"


def _fixture_url(name: str) -> str:
    return (PAGES_DIR / name).as_uri()


def _registry_with_static_browser() -> tuple[object, BrowserSessionManager]:
    manager = BrowserSessionManager(force_static_fallback=True)
    registry = build_default_registry(browser_manager=manager)
    return registry, manager


def test_browser_open_loads_fixture_page() -> None:
    registry, _ = _registry_with_static_browser()

    result = asyncio.run(registry.execute("browser_open", {"url": _fixture_url("example.html")}))

    assert result.status == "success"
    assert result.output["title"] == "Example Domain"
    assert "illustrative examples" in result.output["text_preview"]


def test_browser_read_page_extracts_title_and_text() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("example.html")}))
    result = asyncio.run(registry.execute("browser_read_page", {}))

    assert result.status == "success"
    assert result.output["title"] == "Example Domain"
    assert result.output["text_length"] > 0


def test_browser_extract_links_returns_element_ids() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("example.html")}))
    result = asyncio.run(registry.execute("browser_extract_links", {}))

    assert result.status == "success"
    links = result.output["links"]
    assert links[0]["element_id"] == "link_001"
    assert links[0]["href"].endswith("restaurant.html")


def test_browser_extract_forms_returns_fields() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("booking_form.html")}))
    result = asyncio.run(registry.execute("browser_extract_forms", {}))

    assert result.status == "success"
    forms = result.output["forms"]
    assert forms[0]["form_id"] == "form_001"
    assert [field["element_id"] for field in forms[0]["fields"]] == ["input_001", "input_002", "input_003", "input_004"]
    assert forms[0]["buttons"][0]["risk_hint"] == "high_risk"


def test_browser_click_navigation_link() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("example.html")}))
    result = asyncio.run(registry.execute("browser_click", {"element_id": "link_001"}))

    assert result.status == "success"
    assert result.output["title"] == "Quiet Lantern Restaurant"


def test_browser_fill_safe_text_field() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("booking_form.html")}))
    asyncio.run(registry.execute("browser_extract_forms", {}))
    result = asyncio.run(registry.execute("browser_fill", {"element_id": "input_001", "value": "Test User"}))

    assert result.status == "success"
    assert result.output["form_draft"]["element_id"] == "input_001"
    assert result.output["form_draft"]["value"] == "Test User"


def test_browser_submit_form_requires_permission(tmp_path) -> None:
    registry, manager = _registry_with_static_browser()
    app = create_app(
        database_path=tmp_path / "browser-permission.sqlite3",
        registry=registry,
        browser_manager=manager,
        config=AppConfig(),
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"
            response = test_client.post("/chat", json={"message": "submit this form"})
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
    assert events[2]["data"]["category"] == "LOCAL_TOOL"
    assert events[4]["data"]["tool_name"] == "browser_submit_form"
    assert events[5]["data"]["action_type"] == "browser_submit_form"
    assert app.state.database.count_rows("tool_calls") == 0


def test_payment_page_is_high_risk_or_blocked() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("payment_required.html")}))
    result = asyncio.run(registry.execute("browser_extract_forms", {}))

    form = result.output["forms"][0]
    assert form["fields"][0]["risk_hint"] in {"high_risk", "blocked"}
    assert form["buttons"][0]["risk_hint"] in {"high_risk", "blocked"}


def test_browser_does_not_execute_unknown_element_id() -> None:
    registry, _ = _registry_with_static_browser()

    asyncio.run(registry.execute("browser_open", {"url": _fixture_url("example.html")}))
    result = asyncio.run(registry.execute("browser_click", {"element_id": "link_999"}))

    assert result.status == "failed"
    assert "Unknown browser element id" in result.error


def test_llm_cannot_invent_css_selector() -> None:
    registry, _ = _registry_with_static_browser()

    validation_error = registry.validate_arguments("browser_click", {"selector": "button.reserve"})

    assert validation_error is not None
    assert "Unexpected argument" in validation_error


def test_browser_artifact_created_for_open(tmp_path) -> None:
    registry, manager = _registry_with_static_browser()
    app = create_app(
        database_path=tmp_path / "browser-artifact.sqlite3",
        registry=registry,
        browser_manager=manager,
        config=AppConfig(),
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"
            response = test_client.post("/chat", json={"message": f"open {_fixture_url('example.html')}"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(9)]

    assert "artifact_created" in [event["type"] for event in events]
    artifact_event = next(event for event in events if event["type"] == "artifact_created")
    assert artifact_event["data"]["artifact"]["type"] == "browser_page_snapshot"
    assert artifact_event["data"]["artifact"]["content_text"]
    assert app.state.database.count_rows("artifacts") == 1


def test_browser_task_open_and_read_uses_task_loop_and_artifacts(tmp_path) -> None:
    registry, manager = _registry_with_static_browser()
    app = create_app(
        database_path=tmp_path / "browser-task.sqlite3",
        registry=registry,
        browser_manager=manager,
        config=AppConfig(),
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "event_stream_connected"
            response = test_client.post("/chat", json={"message": f"open {_fixture_url('example.html')} and read the page"})
            assert response.status_code == 200
            events = [websocket.receive_json() for _ in range(14)]

    event_types = [event["type"] for event in events]
    assert "browser_task_started" in event_types
    assert event_types.count("browser_observation_added") == 2
    assert event_types.count("artifact_created") == 2
    assert app.state.database.count_rows("tool_calls") == 2
