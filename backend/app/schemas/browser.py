from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


BrowserElementKind = Literal["link", "button", "input", "select", "textarea"]


class BrowserElement(BaseModel):
    element_id: str
    kind: BrowserElementKind
    text: str | None = None
    label: str | None = None
    href: str | None = None
    risk_hint: str | None = None


class BrowserForm(BaseModel):
    form_id: str
    fields: list[BrowserElement] = Field(default_factory=list)
    buttons: list[BrowserElement] = Field(default_factory=list)


class BrowserPageSnapshot(BaseModel):
    url: str
    title: str = ""
    text_preview: str = ""
    text_length: int = 0
    links: list[BrowserElement] = Field(default_factory=list)
    forms: list[BrowserForm] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BrowserActionPreview(BaseModel):
    action: str
    website: str
    element_id: str | None = None
    form_id: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    risk_level: str
    what_will_happen: str


class BrowserStatusResponse(BaseModel):
    current_url: str | None = None
    page_title: str | None = None
    text_preview: str | None = None
    links: list[BrowserElement] = Field(default_factory=list)
    forms: list[BrowserForm] = Field(default_factory=list)
    last_browser_artifact: dict[str, Any] | None = None
    last_browser_action: str | None = None
    risk_classification: str | None = None
    raw_snapshot: dict[str, Any] | None = None
