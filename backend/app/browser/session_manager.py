from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from urllib.parse import urlparse, urljoin
from urllib.request import urlopen
from urllib.request import url2pathname

from app.browser.element_registry import BrowserElementRegistry
from app.browser.extractors import snapshot_from_html, snapshot_from_playwright_page
from app.browser.safety import normalize_url
from app.schemas.browser import BrowserElement, BrowserForm, BrowserPageSnapshot

try:  # pragma: no cover - exercised only when Playwright is installed with browsers.
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - import fallback is tested through static pages.
    async_playwright = None


LOGGER = logging.getLogger("app.browser.session")


class BrowserSessionError(RuntimeError):
    pass


class BrowserSessionManager:
    """Owns the isolated controlled browser session.

    The real runtime uses Playwright Chromium in a non-persistent context. Tests
    and developer fixtures can still use file/http HTML fallback if Playwright or
    browser binaries are not installed yet.
    """

    def __init__(self, *, headless: bool = False, force_static_fallback: bool = False) -> None:
        self._headless = headless
        self._force_static_fallback = force_static_fallback
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._registry = BrowserElementRegistry()
        self._active_snapshot: BrowserPageSnapshot | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        if self._browser is not None:
            return
        if async_playwright is None:
            raise BrowserSessionError("Playwright is not installed. Install backend dependencies and run 'playwright install chromium'.")

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            await self.stop()
            raise BrowserSessionError(f"Could not start Playwright Chromium: {exc}") from exc

    async def stop(self) -> None:
        for resource in (self._context, self._browser):
            if resource is not None:
                try:
                    await resource.close()
                except Exception:
                    LOGGER.exception("browser resource close failed")
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                LOGGER.exception("playwright stop failed")
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def open_url(self, url: str) -> BrowserPageSnapshot:
        normalized_url = normalize_url(url)
        if await self._can_use_static_fallback(normalized_url):
            self._active_snapshot = await self._open_static_url(normalized_url)
            return self._active_snapshot

        await self.start()
        assert self._page is not None
        try:
            await self._page.goto(normalized_url, wait_until="domcontentloaded", timeout=15000)
            self._active_snapshot = await snapshot_from_playwright_page(self._page, self._registry)
            self._last_error = None
            return self._active_snapshot
        except Exception as exc:
            self._last_error = str(exc)
            raise BrowserSessionError(f"Could not open URL: {exc}") from exc

    async def get_active_page_snapshot(self) -> BrowserPageSnapshot:
        if self._page is not None:
            try:
                self._active_snapshot = await snapshot_from_playwright_page(self._page, self._registry)
                self._last_error = None
                return self._active_snapshot
            except Exception as exc:
                self._last_error = str(exc)
                raise BrowserSessionError(f"Could not read active page: {exc}") from exc
        if self._active_snapshot is not None:
            return self._active_snapshot
        raise BrowserSessionError("No active browser page. Open a URL first.")

    async def extract_links(self) -> list[BrowserElement]:
        snapshot = await self.get_active_page_snapshot()
        return snapshot.links

    async def extract_forms(self) -> list[BrowserForm]:
        snapshot = await self.get_active_page_snapshot()
        return snapshot.forms

    async def click(self, element_id: str) -> BrowserPageSnapshot:
        target = self._registry.get(element_id)
        if target is None:
            raise BrowserSessionError(f"Unknown browser element id: {element_id}")

        if self._page is None:
            if target.href:
                current_url = self._active_snapshot.url if self._active_snapshot else ""
                return await self.open_url(urljoin(current_url, target.href))
            raise BrowserSessionError("Click requires a live browser page for non-link elements.")

        if target.locator_query is None or target.locator_index is None:
            raise BrowserSessionError(f"Element {element_id} is not actionable.")
        try:
            await self._page.locator(target.locator_query).nth(target.locator_index).click(timeout=8000)
            self._active_snapshot = await snapshot_from_playwright_page(self._page, self._registry)
            self._last_error = None
            return self._active_snapshot
        except Exception as exc:
            self._last_error = str(exc)
            raise BrowserSessionError(f"Could not click {element_id}: {exc}") from exc

    async def fill(self, element_id: str, value: str) -> BrowserPageSnapshot:
        target = self._registry.get(element_id)
        if target is None:
            raise BrowserSessionError(f"Unknown browser element id: {element_id}")
        if target.kind not in {"input", "textarea", "select"}:
            raise BrowserSessionError(f"Element {element_id} is not a fillable field.")
        if self._page is None:
            if self._active_snapshot is not None:
                return self._active_snapshot
            raise BrowserSessionError("Fill requires an active browser page.")
        if target.locator_query is None or target.locator_index is None:
            raise BrowserSessionError(f"Element {element_id} is not fillable.")
        try:
            locator = self._page.locator(target.locator_query).nth(target.locator_index)
            if target.kind == "select":
                await locator.select_option(value)
            else:
                await locator.fill(value)
            self._active_snapshot = await snapshot_from_playwright_page(self._page, self._registry)
            self._last_error = None
            return self._active_snapshot
        except Exception as exc:
            self._last_error = str(exc)
            raise BrowserSessionError(f"Could not fill {element_id}: {exc}") from exc

    async def submit_form(self, form_id: str) -> BrowserPageSnapshot:
        if self._page is None:
            raise BrowserSessionError("Submit requires a live browser page.")
        try:
            await self._page.locator(f"form").nth(_form_index(form_id)).evaluate("form => form.requestSubmit()")
            self._active_snapshot = await snapshot_from_playwright_page(self._page, self._registry)
            self._last_error = None
            return self._active_snapshot
        except Exception as exc:
            self._last_error = str(exc)
            raise BrowserSessionError(f"Could not submit {form_id}: {exc}") from exc

    async def screenshot(self) -> dict[str, object]:
        if self._page is None:
            raise BrowserSessionError("Screenshot requires a live browser page.")
        try:
            screenshots_dir = Path(tempfile.gettempdir()) / "vix-browser-screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            path = screenshots_dir / "browser_page.png"
            await self._page.screenshot(path=str(path), full_page=True)
            self._last_error = None
            return {"status": "success", "path": str(path), "source": "playwright"}
        except Exception as exc:
            self._last_error = str(exc)
            raise BrowserSessionError(f"Could not capture browser screenshot: {exc}") from exc

    def get_active_snapshot_if_any(self) -> BrowserPageSnapshot | None:
        return self._active_snapshot

    def dump_element_registry(self) -> dict[str, dict[str, object | None]]:
        return self._registry.dump()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def _can_use_static_fallback(self, url: str) -> bool:
        parsed = urlparse(url)
        return (self._force_static_fallback or async_playwright is None) and parsed.scheme in {"file", "http", "https"}

    async def _open_static_url(self, url: str) -> BrowserPageSnapshot:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            path = Path(url2pathname(parsed.path))
            if parsed.netloc:
                path = Path(url2pathname(f"//{parsed.netloc}{parsed.path}"))
            html = path.read_text(encoding="utf-8")
        else:
            with urlopen(url, timeout=10) as response:  # noqa: S310 - user explicitly controls Stage 5 browser URL.
                html = response.read().decode("utf-8", errors="replace")
        snapshot = snapshot_from_html(url, html, self._registry)
        self._last_error = None
        return snapshot


def _form_index(form_id: str) -> int:
    try:
        return max(0, int(form_id.split("_", 1)[1]) - 1)
    except Exception as exc:
        raise BrowserSessionError(f"Invalid form id: {form_id}") from exc
