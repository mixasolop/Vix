from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class WindowInfo(BaseModel):
    hwnd: int
    title: str = ""
    process_id: int | None = None
    process_name: str | None = None
    executable_path: str | None = None
    is_vix: bool = False
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WindowContextSnapshot(BaseModel):
    current_foreground_window: WindowInfo | None = None
    last_non_vix_window: WindowInfo | None = None
    last_context_window: WindowInfo | None = None
    last_context_captured_at: datetime | None = None


class SelectedTextResult(BaseModel):
    status: str
    text: str = ""
    method: str = "context_window_ctrl_c"
    context_window: WindowInfo | None = None
    restored_clipboard: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextStatusResponse(BaseModel):
    current_foreground_window: WindowInfo | None = None
    last_context_window: WindowInfo | None = None
    last_context_artifact: dict[str, Any] | None = None
