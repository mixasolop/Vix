from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.browser.safety import risk_hint_for_text
from app.schemas.browser import BrowserElement


@dataclass(frozen=True)
class BrowserElementTarget:
    element_id: str
    kind: str
    locator_query: str | None = None
    locator_index: int | None = None
    href: str | None = None
    form_id: str | None = None


class BrowserElementRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, BrowserElementTarget] = {}
        self._counters: dict[str, int] = {}

    def reset(self) -> None:
        self._targets.clear()
        self._counters.clear()

    def register(
        self,
        *,
        kind: Literal["link", "button", "input", "select", "textarea"],
        text: str | None = None,
        label: str | None = None,
        href: str | None = None,
        locator_query: str | None = None,
        locator_index: int | None = None,
        form_id: str | None = None,
    ) -> BrowserElement:
        number = self._counters.get(kind, 0) + 1
        self._counters[kind] = number
        element_id = f"{kind}_{number:03d}"
        self._targets[element_id] = BrowserElementTarget(
            element_id=element_id,
            kind=kind,
            locator_query=locator_query,
            locator_index=locator_index,
            href=href,
            form_id=form_id,
        )
        return BrowserElement(
            element_id=element_id,
            kind=kind,
            text=_clean(text),
            label=_clean(label),
            href=_clean(href),
            risk_hint=risk_hint_for_text(" ".join(part for part in (text or "", label or "") if part), href),
        )

    def get(self, element_id: str) -> BrowserElementTarget | None:
        return self._targets.get(element_id)

    def dump(self) -> dict[str, dict[str, object | None]]:
        return {
            element_id: {
                "kind": target.kind,
                "locator_query": target.locator_query,
                "locator_index": target.locator_index,
                "href": target.href,
                "form_id": target.form_id,
            }
            for element_id, target in self._targets.items()
        }


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None
