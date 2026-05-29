from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
import ctypes
from ctypes import wintypes
import logging
import os
import threading
import time

from app.schemas.window_context import WindowContextSnapshot, WindowInfo

LOGGER = logging.getLogger("app.context.window_tracker")

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
EVENT_SYSTEM_FOREGROUND = 0x0003
WINEVENT_OUTOFCONTEXT = 0x0000
GW_HWNDNEXT = 2
SW_RESTORE = 9
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002


class WindowTracker:
    def __init__(self, poll_interval_seconds: float = 0.5) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = threading.RLock()
        self._current_foreground_window: WindowInfo | None = None
        self._last_non_vix_window: WindowInfo | None = None
        self._last_context_window: WindowInfo | None = None
        self._last_context_captured_at: datetime | None = None
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._hook_handle: int | None = None
        self._win_event_proc = None
        self._on_context_window_updated: Callable[[WindowInfo], None] | None = None

    def set_context_updated_callback(self, callback: Callable[[WindowInfo], None] | None) -> None:
        self._on_context_window_updated = callback

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self.refresh_foreground_window()
        self._install_foreground_hook()
        self._poll_thread = threading.Thread(target=self._poll_foreground_loop, name="vix-window-tracker", daemon=True)
        self._poll_thread.start()
        LOGGER.info("window tracker started | hook_installed=%s", self._hook_handle is not None)

    def stop(self) -> None:
        self._running = False
        self._uninstall_foreground_hook()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
            self._poll_thread = None
        LOGGER.info("window tracker stopped")

    def refresh_foreground_window(self) -> WindowInfo | None:
        hwnd = _get_foreground_hwnd()
        if not hwnd:
            return None
        info = read_window_info(hwnd)
        self.on_foreground_window_changed(info)
        return info

    def on_foreground_window_changed(self, info: WindowInfo | None) -> None:
        if info is None:
            return

        callback_info: WindowInfo | None = None
        with self._lock:
            previous_context_hwnd = self._last_context_window.hwnd if self._last_context_window else None
            self._current_foreground_window = info
            if not info.is_vix:
                self._last_non_vix_window = info
                self._last_context_window = info
                self._last_context_captured_at = info.captured_at
                if previous_context_hwnd != info.hwnd:
                    callback_info = info

        if callback_info is not None and self._on_context_window_updated is not None:
            try:
                self._on_context_window_updated(callback_info)
            except Exception:
                LOGGER.exception("context window update callback failed")

    def get_current_foreground_window(self, refresh: bool = True) -> WindowInfo | None:
        if refresh:
            self.refresh_foreground_window()
        with self._lock:
            return self._current_foreground_window

    def get_context_window(self, validate_exists: bool = False) -> WindowInfo | None:
        with self._lock:
            info = self._last_context_window
        if info is None or (validate_exists and not window_exists(info.hwnd)):
            info = self._find_fallback_context_window()
            if info is not None:
                self._set_context_window(info)
        if info is None:
            return None
        if validate_exists and not window_exists(info.hwnd):
            return None
        return info

    def snapshot(self) -> WindowContextSnapshot:
        with self._lock:
            return WindowContextSnapshot(
                current_foreground_window=self._current_foreground_window,
                last_non_vix_window=self._last_non_vix_window,
                last_context_window=self._last_context_window,
                last_context_captured_at=self._last_context_captured_at,
            )

    def _install_foreground_hook(self) -> None:
        if os.name != "nt":
            return

        try:
            WinEventProcType = ctypes.WINFUNCTYPE(
                None,
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.HWND,
                wintypes.LONG,
                wintypes.LONG,
                wintypes.DWORD,
                wintypes.DWORD,
            )

            def callback(hook, event, hwnd, object_id, child_id, event_thread, event_time):
                if event == EVENT_SYSTEM_FOREGROUND and hwnd:
                    self.on_foreground_window_changed(read_window_info(int(hwnd)))

            self._win_event_proc = WinEventProcType(callback)
            handle = ctypes.windll.user32.SetWinEventHook(
                EVENT_SYSTEM_FOREGROUND,
                EVENT_SYSTEM_FOREGROUND,
                0,
                self._win_event_proc,
                0,
                0,
                WINEVENT_OUTOFCONTEXT,
            )
            self._hook_handle = int(handle) if handle else None
        except Exception:
            self._hook_handle = None
            LOGGER.exception("failed to install foreground window hook; polling fallback remains active")

    def _uninstall_foreground_hook(self) -> None:
        if os.name != "nt" or self._hook_handle is None:
            return
        try:
            ctypes.windll.user32.UnhookWinEvent(self._hook_handle)
        except Exception:
            LOGGER.exception("failed to uninstall foreground window hook")
        finally:
            self._hook_handle = None
            self._win_event_proc = None

    def _poll_foreground_loop(self) -> None:
        while self._running:
            try:
                self.refresh_foreground_window()
            except Exception:
                LOGGER.exception("foreground polling failed")
            time.sleep(self._poll_interval_seconds)

    def _set_context_window(self, info: WindowInfo) -> None:
        callback_info: WindowInfo | None = None
        with self._lock:
            previous_context_hwnd = self._last_context_window.hwnd if self._last_context_window else None
            self._last_non_vix_window = info
            self._last_context_window = info
            self._last_context_captured_at = info.captured_at
            if previous_context_hwnd != info.hwnd:
                callback_info = info

        if callback_info is not None and self._on_context_window_updated is not None:
            try:
                self._on_context_window_updated(callback_info)
            except Exception:
                LOGGER.exception("context window update callback failed")

    def _find_fallback_context_window(self) -> WindowInfo | None:
        info = find_top_non_vix_window()
        if info is not None:
            LOGGER.info(
                "fallback context window selected | hwnd=%s | title=%s | process=%s",
                info.hwnd,
                info.title,
                info.process_name,
            )
        return info


def read_window_info(hwnd: int) -> WindowInfo:
    title = _get_window_title(hwnd)
    process_id = _get_window_process_id(hwnd)
    executable_path = _get_process_executable_path(process_id) if process_id else None
    process_name = Path(executable_path).name if executable_path else None
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        process_id=process_id,
        process_name=process_name,
        executable_path=executable_path,
        is_vix=is_vix_window(title=title, process_name=process_name, executable_path=executable_path),
        captured_at=datetime.now(UTC),
    )


def is_vix_window(title: str, process_name: str | None, executable_path: str | None = None) -> bool:
    process = (process_name or "").lower()
    path = (executable_path or "").lower()
    normalized_title = title.strip().lower()
    if process in {"desktopassistant.frontend.exe", "desktopassistant.exe"}:
        return True
    if "desktopassistant.frontend" in process:
        return True
    if normalized_title == "desktop assistant":
        return True
    return "frontend-wpf" in path and "desktopassistant" in path


def window_exists(hwnd: int) -> bool:
    if os.name != "nt" or not hwnd:
        return False
    return bool(ctypes.windll.user32.IsWindow(wintypes.HWND(hwnd)))


def activate_window(hwnd: int) -> bool:
    if os.name != "nt" or not hwnd or not window_exists(hwnd):
        return False
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        _configure_activation_api(user32, kernel32)

        if _foreground_is(hwnd):
            return True

        hwnd_handle = wintypes.HWND(hwnd)
        user32.ShowWindow(hwnd_handle, SW_RESTORE)
        user32.SetWindowPos(hwnd_handle, wintypes.HWND(0), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        _tap_alt_key(user32)
        user32.BringWindowToTop(hwnd_handle)
        user32.SetForegroundWindow(hwnd_handle)
        if _wait_for_foreground(hwnd):
            return True

        current_hwnd = user32.GetForegroundWindow()
        current_thread = user32.GetWindowThreadProcessId(wintypes.HWND(current_hwnd), None) if current_hwnd else 0
        target_thread = user32.GetWindowThreadProcessId(hwnd_handle, None)
        this_thread = kernel32.GetCurrentThreadId()
        attached_current = False
        attached_self = False
        try:
            if target_thread and current_thread and current_thread != target_thread:
                attached_current = bool(user32.AttachThreadInput(current_thread, target_thread, True))
            if target_thread and this_thread != target_thread:
                attached_self = bool(user32.AttachThreadInput(this_thread, target_thread, True))
            user32.ShowWindow(hwnd_handle, SW_RESTORE)
            user32.BringWindowToTop(hwnd_handle)
            user32.SetForegroundWindow(hwnd_handle)
            user32.SetFocus(hwnd_handle)
            return _wait_for_foreground(hwnd)
        finally:
            if attached_self:
                user32.AttachThreadInput(this_thread, target_thread, False)
            if attached_current:
                user32.AttachThreadInput(current_thread, target_thread, False)
    except Exception:
        LOGGER.exception("failed to activate window | hwnd=%s", hwnd)
        return False


def find_top_non_vix_window() -> WindowInfo | None:
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    hwnd = user32.GetTopWindow(None)
    while hwnd:
        hwnd_int = int(hwnd)
        if _is_user_context_candidate(hwnd_int):
            info = read_window_info(hwnd_int)
            if not info.is_vix:
                return info
        hwnd = user32.GetWindow(wintypes.HWND(hwnd_int), GW_HWNDNEXT)
    return None


def _is_user_context_candidate(hwnd: int) -> bool:
    if os.name != "nt" or not hwnd:
        return False
    user32 = ctypes.windll.user32
    if not user32.IsWindowVisible(wintypes.HWND(hwnd)):
        return False
    if user32.IsIconic(wintypes.HWND(hwnd)):
        return False
    title = _get_window_title(hwnd).strip()
    if not title:
        return False
    if title.lower() in {"program manager", "windows input experience"}:
        return False
    return True


def _get_foreground_hwnd() -> int | None:
    if os.name != "nt":
        return None
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def _foreground_is(hwnd: int) -> bool:
    foreground = _get_foreground_hwnd()
    return foreground == hwnd


def _wait_for_foreground(hwnd: int, timeout_seconds: float = 0.6) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _foreground_is(hwnd):
            return True
        time.sleep(0.03)
    return _foreground_is(hwnd)


def _tap_alt_key(user32) -> None:
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def _configure_activation_api(user32, kernel32) -> None:
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetFocus.argtypes = [wintypes.HWND]
    user32.SetFocus.restype = wintypes.HWND
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.AttachThreadInput.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, wintypes.ULONG]
    user32.keybd_event.restype = None
    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD


def _get_window_title(hwnd: int) -> str:
    if os.name != "nt" or not hwnd:
        return ""
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1)
    return buffer.value


def _get_window_process_id(hwnd: int) -> int | None:
    if os.name != "nt" or not hwnd:
        return None
    process_id = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(process_id))
    return int(process_id.value) if process_id.value else None


def _get_process_executable_path(process_id: int) -> str | None:
    if os.name != "nt" or not process_id:
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return None
    finally:
        kernel32.CloseHandle(handle)
