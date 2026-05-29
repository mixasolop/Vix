from collections.abc import Awaitable, Callable
import asyncio
import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import logging
import os
import time
from typing import Protocol

from app.context.window_tracker import WindowTracker, activate_window, window_exists
from app.schemas.tools import ToolResult
from app.schemas.window_context import SelectedTextResult, WindowInfo

LOGGER = logging.getLogger("app.tools.context")

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_C = 0x43
KEYEVENTF_KEYUP = 0x0002
WM_COPY = 0x0301
SMTO_ABORTIFHUNG = 0x0002

ContextExecutor = Callable[[dict[str, object]], Awaitable[ToolResult]]


class SelectedTextCaptureStrategy(Protocol):
    async def capture(self, target: WindowInfo) -> SelectedTextResult:
        ...


class ClipboardCopyStrategy:
    async def capture(self, target: WindowInfo) -> SelectedTextResult:
        return await asyncio.to_thread(self._capture_sync, target)

    def _capture_sync(self, target: WindowInfo) -> SelectedTextResult:
        if target.is_vix:
            return SelectedTextResult(status="failed", context_window=target, error="Context window is Vix.")
        if not window_exists(target.hwnd):
            return SelectedTextResult(status="failed", context_window=target, error="Context window no longer exists.")

        previous_foreground_hwnd = _get_foreground_hwnd()
        previous_clipboard = _safe_read_clipboard_text()
        previous_text = previous_clipboard.text if previous_clipboard.success else None
        previous_had_text = previous_clipboard.success
        if not activate_window(target.hwnd):
            fallback = _copy_selection_with_wm_copy(target, previous_text, previous_had_text)
            if fallback.status == "success":
                return fallback
            return SelectedTextResult(
                status="failed",
                context_window=target,
                restored_clipboard=False,
                error="Could not focus context window.",
                metadata={
                    "clipboard_read_error": previous_clipboard.error,
                    "wm_copy_error": fallback.error,
                    "wm_copy_metadata": fallback.metadata,
                },
            )

        _send_copy_hotkey()
        time.sleep(0.2)
        captured = _safe_read_clipboard_text()
        if not captured.success or not captured.text.strip():
            _send_copy_hotkey(use_shift=True)
            time.sleep(0.2)
            captured = _safe_read_clipboard_text()
        restored_clipboard = _safe_restore_clipboard(previous_text, previous_had_text)
        restored_focus = _restore_focus(previous_foreground_hwnd, target.hwnd)
        if not captured.success:
            return SelectedTextResult(
                status="failed",
                context_window=target,
                restored_clipboard=restored_clipboard,
                error=captured.error or "Ctrl+C produced no text.",
                metadata={"restored_focus": restored_focus},
            )

        text = captured.text.strip()
        if not text:
            return SelectedTextResult(
                status="failed",
                context_window=target,
                restored_clipboard=restored_clipboard,
                error="Ctrl+C produced no text.",
                metadata={"restored_focus": restored_focus},
            )

        return SelectedTextResult(
            status="success",
            text=text,
            context_window=target,
            restored_clipboard=restored_clipboard,
            metadata={
                "clipboard_restore_error": None if restored_clipboard else "Clipboard could not be restored.",
                "restored_focus": restored_focus,
            },
        )


def build_context_tool_executors(
    tracker: WindowTracker,
    selected_text_strategy: SelectedTextCaptureStrategy | None = None,
) -> dict[str, ContextExecutor]:
    strategy = selected_text_strategy or ClipboardCopyStrategy()

    async def get_foreground_window_info(arguments: dict[str, object]) -> ToolResult:
        window = tracker.get_current_foreground_window(refresh=True)
        if window is None:
            return ToolResult(tool="get_foreground_window_info", status="failed", error="No foreground window was captured.")
        return ToolResult(
            tool="get_foreground_window_info",
            status="success",
            output={
                "status": "success",
                "message": _window_message("Foreground window", window),
                "window": window.model_dump(mode="json"),
                "source": "foreground_window",
            },
        )

    async def get_context_window_info(arguments: dict[str, object]) -> ToolResult:
        window = tracker.get_context_window(validate_exists=True)
        if window is None:
            return ToolResult(tool="get_context_window_info", status="failed", error="No previous non-Vix window was captured.")
        return ToolResult(
            tool="get_context_window_info",
            status="success",
            output={
                "status": "success",
                "message": _window_message("Context window", window),
                "window": window.model_dump(mode="json"),
                "source": "context_window",
            },
        )

    async def get_clipboard_text(arguments: dict[str, object]) -> ToolResult:
        result = await asyncio.to_thread(_safe_read_clipboard_text)
        if not result.success:
            return ToolResult(tool="get_clipboard_text", status="failed", error=result.error or "Clipboard text could not be read.")
        return ToolResult(
            tool="get_clipboard_text",
            status="success",
            output={
                "status": "success",
                "message": "Clipboard text captured.",
                "text": result.text,
                "length": len(result.text),
                "source": "clipboard",
                "captured_at": datetime.now(UTC).isoformat(),
            },
        )

    async def get_selected_text(arguments: dict[str, object]) -> ToolResult:
        window = tracker.get_context_window(validate_exists=True)
        if window is None:
            return ToolResult(tool="get_selected_text", status="failed", error="No previous non-Vix window was captured.")
        if window.is_vix:
            return ToolResult(tool="get_selected_text", status="failed", error="Context window is Vix.")

        result = await strategy.capture(window)
        if result.status != "success":
            return ToolResult(
                tool="get_selected_text",
                status="failed",
                output={
                    "method": result.method,
                    "context_window": window.model_dump(mode="json"),
                    "restored_clipboard": result.restored_clipboard,
                    "metadata": result.metadata,
                },
                error=result.error or "Selected text could not be captured.",
            )

        return ToolResult(
            tool="get_selected_text",
            status="success",
            output={
                "status": "success",
                "message": "Selected text captured.",
                "text": result.text,
                "length": len(result.text),
                "method": result.method,
                "context_window": window.model_dump(mode="json"),
                "restored_clipboard": result.restored_clipboard,
                "metadata": result.metadata,
                "captured_at": datetime.now(UTC).isoformat(),
            },
        )

    return {
        "get_foreground_window_info": get_foreground_window_info,
        "get_context_window_info": get_context_window_info,
        "get_clipboard_text": get_clipboard_text,
        "get_selected_text": get_selected_text,
    }


class _ClipboardReadResult:
    def __init__(self, success: bool, text: str = "", error: str | None = None) -> None:
        self.success = success
        self.text = text
        self.error = error


def _window_message(prefix: str, window: WindowInfo) -> str:
    label = window.process_name or window.title or str(window.hwnd)
    return f"{prefix} is {label}."


def _safe_read_clipboard_text() -> _ClipboardReadResult:
    try:
        return _try_read_clipboard_text()
    except Exception as exc:
        LOGGER.exception("clipboard read failed unexpectedly")
        return _ClipboardReadResult(False, error=f"Clipboard read failed: {exc}")


def _safe_restore_clipboard(text: str | None, had_text: bool) -> bool:
    try:
        return _restore_clipboard(text, had_text)
    except Exception:
        LOGGER.exception("clipboard restore failed unexpectedly")
        return False


def _copy_selection_with_wm_copy(target: WindowInfo, previous_text: str | None, previous_had_text: bool) -> SelectedTextResult:
    sentinel = f"__VIX_CLIPBOARD_SENTINEL_{time.time_ns()}__"
    if not _safe_restore_clipboard(sentinel, had_text=True):
        return SelectedTextResult(
            status="failed",
            context_window=target,
            error="Could not prepare clipboard for WM_COPY fallback.",
            metadata={"method_fallback": "wm_copy"},
        )

    copied_from: int | None = None
    for candidate_hwnd in _copy_message_candidates(target.hwnd):
        if not _send_wm_copy(candidate_hwnd):
            continue
        time.sleep(0.08)
        captured = _safe_read_clipboard_text()
        if captured.success:
            text = captured.text.strip()
            if text and text != sentinel:
                copied_from = candidate_hwnd
                restored = _safe_restore_clipboard(previous_text, previous_had_text)
                return SelectedTextResult(
                    status="success",
                    text=text,
                    context_window=target,
                    restored_clipboard=restored,
                    metadata={
                        "method_fallback": "wm_copy",
                        "wm_copy_hwnd": copied_from,
                        "clipboard_restore_error": None if restored else "Clipboard could not be restored.",
                    },
                )

    restored = _safe_restore_clipboard(previous_text, previous_had_text)
    return SelectedTextResult(
        status="failed",
        context_window=target,
        restored_clipboard=restored,
        error="WM_COPY fallback produced no selected text.",
        metadata={"method_fallback": "wm_copy", "restored_clipboard": restored},
    )


def _try_read_clipboard_text() -> _ClipboardReadResult:
    if os.name != "nt":
        return _ClipboardReadResult(False, error="Clipboard access is only implemented on Windows.")
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    _configure_clipboard_api(user32, kernel32)
    if not user32.OpenClipboard(None):
        return _ClipboardReadResult(False, error="Clipboard is locked or unavailable.")
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return _ClipboardReadResult(False, error="Clipboard does not contain text.")
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return _ClipboardReadResult(False, error="Clipboard text handle was empty.")
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return _ClipboardReadResult(False, error="Clipboard text could not be locked.")
        try:
            return _ClipboardReadResult(True, ctypes.wstring_at(pointer))
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _restore_clipboard(text: str | None, had_text: bool) -> bool:
    if os.name != "nt":
        return False
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    _configure_clipboard_api(user32, kernel32)
    if not user32.OpenClipboard(None):
        return False
    try:
        if not user32.EmptyClipboard():
            return False
        if not had_text:
            return True
        text = text or ""
        encoded_size = (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, encoded_size)
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return False
        try:
            buffer = ctypes.create_unicode_buffer(text)
            ctypes.memmove(pointer, ctypes.addressof(buffer), encoded_size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            return False
        return True
    finally:
        user32.CloseClipboard()


def _send_copy_hotkey(use_shift: bool = False) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    if use_shift:
        user32.keybd_event(VK_SHIFT, 0, 0, 0)
    user32.keybd_event(VK_C, 0, 0, 0)
    user32.keybd_event(VK_C, 0, KEYEVENTF_KEYUP, 0)
    if use_shift:
        user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _get_foreground_hwnd() -> int | None:
    if os.name != "nt":
        return None
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def _restore_focus(previous_hwnd: int | None, target_hwnd: int) -> bool | None:
    if previous_hwnd is None or previous_hwnd == target_hwnd:
        return None
    return activate_window(previous_hwnd)


def _copy_message_candidates(hwnd: int) -> list[int]:
    if os.name != "nt" or not hwnd:
        return []
    candidates = [hwnd]
    candidates.extend(_child_windows(hwnd))
    return candidates


def _child_windows(hwnd: int) -> list[int]:
    if os.name != "nt" or not hwnd:
        return []

    user32 = ctypes.windll.user32
    child_windows: list[int] = []
    EnumChildProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(child_hwnd, lparam):
        child_windows.append(int(child_hwnd))
        return True

    enum_proc = EnumChildProc(callback)
    user32.EnumChildWindows.argtypes = [wintypes.HWND, EnumChildProc, wintypes.LPARAM]
    user32.EnumChildWindows.restype = wintypes.BOOL
    user32.EnumChildWindows(wintypes.HWND(hwnd), enum_proc, 0)
    return child_windows


def _send_wm_copy(hwnd: int) -> bool:
    if os.name != "nt" or not hwnd:
        return False

    user32 = ctypes.windll.user32
    result = ctypes.c_size_t()
    user32.SendMessageTimeoutW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.SendMessageTimeoutW.restype = wintypes.LPARAM
    return bool(
        user32.SendMessageTimeoutW(
            wintypes.HWND(hwnd),
            WM_COPY,
            0,
            0,
            SMTO_ABORTIFHUNG,
            200,
            ctypes.byref(result),
        )
    )


def _configure_clipboard_api(user32, kernel32) -> None:
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
