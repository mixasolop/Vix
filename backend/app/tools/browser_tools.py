from __future__ import annotations

from collections.abc import Awaitable, Callable
from urllib.parse import quote_plus

from app.browser.page_snapshot import snapshot_message, snapshot_summary
from app.browser.safety import assess_browser_text, normalize_url
from app.browser.session_manager import BrowserSessionError, BrowserSessionManager
from app.schemas.browser import BrowserPageSnapshot
from app.schemas.tools import ToolResult


BrowserToolExecutor = Callable[[dict[str, object]], Awaitable[ToolResult]]


def build_browser_tool_executors(manager: BrowserSessionManager) -> dict[str, BrowserToolExecutor]:
    async def browser_open(arguments: dict[str, object]) -> ToolResult:
        url = str(arguments.get("url") or "")
        try:
            snapshot = await manager.open_url(normalize_url(url))
        except (ValueError, BrowserSessionError) as exc:
            return ToolResult(tool="browser_open", status="failed", error=str(exc))
        return _snapshot_result("browser_open", snapshot, message=f"Opened {snapshot.url}. {snapshot_message(snapshot)}")

    async def browser_read_page(arguments: dict[str, object]) -> ToolResult:
        try:
            snapshot = await manager.get_active_page_snapshot()
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_read_page", status="failed", error=str(exc))
        return _snapshot_result("browser_read_page", snapshot, message=snapshot_message(snapshot))

    async def browser_extract_links(arguments: dict[str, object]) -> ToolResult:
        try:
            links = await manager.extract_links()
            snapshot = await manager.get_active_page_snapshot()
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_extract_links", status="failed", error=str(exc))
        return ToolResult(
            tool="browser_extract_links",
            status="success",
            output={
                "status": "success",
                "message": f"Found {len(links)} links on {snapshot.title or snapshot.url}.",
                "url": snapshot.url,
                "title": snapshot.title,
                "links": [link.model_dump(mode="json") for link in links],
                "snapshot": snapshot.model_dump(mode="json"),
            },
        )

    async def browser_extract_forms(arguments: dict[str, object]) -> ToolResult:
        try:
            forms = await manager.extract_forms()
            snapshot = await manager.get_active_page_snapshot()
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_extract_forms", status="failed", error=str(exc))
        return ToolResult(
            tool="browser_extract_forms",
            status="success",
            output={
                "status": "success",
                "message": f"Found {len(forms)} forms on {snapshot.title or snapshot.url}.",
                "url": snapshot.url,
                "title": snapshot.title,
                "forms": [form.model_dump(mode="json") for form in forms],
                "snapshot": snapshot.model_dump(mode="json"),
            },
        )

    async def browser_search(arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(tool="browser_search", status="failed", error="query is required.")
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            snapshot = await manager.open_url(search_url)
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_search", status="failed", error=str(exc))
        return ToolResult(
            tool="browser_search",
            status="success",
            output={
                "status": "success",
                "message": f"Opened search results for '{query}' and found {len(snapshot.links)} candidate links.",
                "query": query,
                "url": snapshot.url,
                "title": snapshot.title,
                "results": [link.model_dump(mode="json") for link in snapshot.links[:10]],
                "snapshot": snapshot.model_dump(mode="json"),
            },
        )

    async def browser_screenshot(arguments: dict[str, object]) -> ToolResult:
        try:
            screenshot = await manager.screenshot()
            snapshot = manager.get_active_snapshot_if_any()
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_screenshot", status="failed", error=str(exc))
        return ToolResult(
            tool="browser_screenshot",
            status="success",
            output={
                "status": "success",
                "message": "Browser screenshot captured.",
                "screenshot": screenshot,
                "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
            },
        )

    async def browser_click(arguments: dict[str, object]) -> ToolResult:
        element_id = str(arguments.get("element_id") or "")
        if not element_id:
            return ToolResult(tool="browser_click", status="failed", error="element_id is required.")
        try:
            snapshot = await manager.click(element_id)
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_click", status="failed", error=str(exc))
        return _snapshot_result("browser_click", snapshot, message=f"Clicked {element_id}. {snapshot_message(snapshot)}")

    async def browser_fill(arguments: dict[str, object]) -> ToolResult:
        element_id = str(arguments.get("element_id") or "")
        value = str(arguments.get("value") or "")
        assessment = assess_browser_text(f"{element_id} {value}")
        if assessment.blocked:
            return ToolResult(tool="browser_fill", status="failed", error=assessment.reason)
        if not element_id:
            return ToolResult(tool="browser_fill", status="failed", error="element_id is required.")
        try:
            snapshot = await manager.fill(element_id, value)
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_fill", status="failed", error=str(exc))
        return _snapshot_result(
            "browser_fill",
            snapshot,
            message=f"Filled {element_id}. No form was submitted.",
            extra={"form_draft": {"element_id": element_id, "value": value, "risk_level": assessment.risk_level.value}},
        )

    async def browser_submit_form(arguments: dict[str, object]) -> ToolResult:
        form_id = str(arguments.get("form_id") or "")
        if not form_id:
            return ToolResult(tool="browser_submit_form", status="failed", error="form_id is required.")
        try:
            snapshot = await manager.submit_form(form_id)
        except BrowserSessionError as exc:
            return ToolResult(tool="browser_submit_form", status="failed", error=str(exc))
        return _snapshot_result("browser_submit_form", snapshot, message=f"Submitted {form_id}. {snapshot_message(snapshot)}")

    return {
        "browser_open": browser_open,
        "browser_read_page": browser_read_page,
        "browser_extract_links": browser_extract_links,
        "browser_extract_forms": browser_extract_forms,
        "browser_search": browser_search,
        "browser_screenshot": browser_screenshot,
        "browser_click": browser_click,
        "browser_fill": browser_fill,
        "browser_submit_form": browser_submit_form,
    }


def _snapshot_result(tool_name: str, snapshot: BrowserPageSnapshot, *, message: str, extra: dict[str, object] | None = None) -> ToolResult:
    output: dict[str, object] = {
        "status": "success",
        "message": message,
        "snapshot": snapshot.model_dump(mode="json"),
        **snapshot_summary(snapshot),
    }
    if extra:
        output.update(extra)
    return ToolResult(tool=tool_name, status="success", output=output)
