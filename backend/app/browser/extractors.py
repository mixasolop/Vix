from __future__ import annotations

from html.parser import HTMLParser
import re
from urllib.parse import urljoin

from app.browser.element_registry import BrowserElementRegistry
from app.schemas.browser import BrowserForm, BrowserPageSnapshot


MAX_LINKS = 80
MAX_FORMS = 20
MAX_TEXT_PREVIEW = 5000


async def snapshot_from_playwright_page(page, registry: BrowserElementRegistry) -> BrowserPageSnapshot:
    registry.reset()
    payload = await page.evaluate(
        """
        () => {
          const text = document.body ? document.body.innerText : "";
          const title = document.title || "";
          const linkNodes = Array.from(document.querySelectorAll("a"));
          const controlNodes = Array.from(document.querySelectorAll("input, select, textarea, button"));
          const formNodes = Array.from(document.querySelectorAll("form"));
          const controlIndex = new Map(controlNodes.map((node, index) => [node, index]));
          const describe = (node) => ({
            tag: node.tagName.toLowerCase(),
            type: node.getAttribute("type") || "",
            text: (node.innerText || node.value || "").trim(),
            label: (node.getAttribute("aria-label") || node.getAttribute("placeholder") || node.getAttribute("name") || node.id || "").trim(),
            href: node.href || node.getAttribute("href") || "",
          });
          return {
            url: window.location.href,
            title,
            text,
            links: linkNodes.slice(0, 80).map((node, index) => ({...describe(node), index})),
            forms: formNodes.slice(0, 20).map((form, formIndex) => {
              const controls = Array.from(form.querySelectorAll("input, select, textarea, button"));
              return {
                formIndex,
                controls: controls.map((node) => ({...describe(node), controlIndex: controlIndex.get(node)}))
              };
            }),
          };
        }
        """
    )
    return snapshot_from_payload(payload, registry)


def snapshot_from_html(url: str, html: str, registry: BrowserElementRegistry) -> BrowserPageSnapshot:
    registry.reset()
    parser = _SnapshotHtmlParser(base_url=url)
    parser.feed(html)
    payload = parser.to_payload()
    payload["url"] = url
    return snapshot_from_payload(payload, registry)


def snapshot_from_payload(payload: dict[str, object], registry: BrowserElementRegistry) -> BrowserPageSnapshot:
    text = str(payload.get("text") or "")
    title = str(payload.get("title") or "")
    links = []
    for link in list(payload.get("links") or [])[:MAX_LINKS]:
        if not isinstance(link, dict):
            continue
        links.append(
            registry.register(
                kind="link",
                text=str(link.get("text") or ""),
                label=str(link.get("label") or ""),
                href=str(link.get("href") or ""),
                locator_query="a",
                locator_index=int(link.get("index") or 0),
            )
        )

    forms: list[BrowserForm] = []
    for form_index, form in enumerate(list(payload.get("forms") or [])[:MAX_FORMS], start=1):
        if not isinstance(form, dict):
            continue
        form_id = f"form_{form_index:03d}"
        fields = []
        buttons = []
        for control in list(form.get("controls") or []):
            if not isinstance(control, dict):
                continue
            tag = str(control.get("tag") or "").lower()
            input_type = str(control.get("type") or "").lower()
            if tag == "button" or input_type in {"submit", "button"}:
                kind = "button"
            elif tag in {"select", "textarea"}:
                kind = tag
            else:
                kind = "input"
            element = registry.register(
                kind=kind,
                text=str(control.get("text") or ""),
                label=str(control.get("label") or ""),
                href=str(control.get("href") or ""),
                locator_query="input, select, textarea, button",
                locator_index=int(control.get("controlIndex") or control.get("index") or 0),
                form_id=form_id,
            )
            if kind == "button":
                buttons.append(element)
            else:
                fields.append(element)
        forms.append(BrowserForm(form_id=form_id, fields=fields, buttons=buttons))

    return BrowserPageSnapshot(
        url=str(payload.get("url") or ""),
        title=title,
        text_preview=_preview_text(text),
        text_length=len(text),
        links=links,
        forms=forms,
    )


class _SnapshotHtmlParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._title_chunks: list[str] = []
        self._text_chunks: list[str] = []
        self._links: list[dict[str, object]] = []
        self._forms: list[dict[str, object]] = []
        self._tag_stack: list[str] = []
        self._current_link: dict[str, object] | None = None
        self._current_form: dict[str, object] | None = None
        self._current_button: dict[str, object] | None = None
        self._control_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = {name.lower(): value or "" for name, value in attrs}
        self._tag_stack.append(tag)
        if tag == "a":
            self._current_link = {"tag": "a", "text": "", "label": _label(attrs_map), "href": urljoin(self._base_url, attrs_map.get("href", "")), "index": len(self._links)}
        elif tag == "form":
            self._current_form = {"formIndex": len(self._forms), "controls": []}
        elif tag in {"input", "select", "textarea", "button"}:
            control = {
                "tag": tag,
                "type": attrs_map.get("type", ""),
                "text": attrs_map.get("value", ""),
                "label": _label(attrs_map),
                "href": "",
                "controlIndex": self._control_index,
            }
            self._control_index += 1
            if tag == "button":
                self._current_button = control
            if self._current_form is not None:
                self._current_form["controls"].append(control)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._current_link is not None:
            self._links.append(self._current_link)
            self._current_link = None
        elif tag == "form" and self._current_form is not None:
            self._forms.append(self._current_form)
            self._current_form = None
        elif tag == "button":
            self._current_button = None
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self._title_chunks.append(text)
        if self._current_link is not None:
            self._current_link["text"] = f"{self._current_link.get('text', '')} {text}".strip()
        if self._current_button is not None:
            self._current_button["text"] = f"{self._current_button.get('text', '')} {text}".strip()
        if not self._tag_stack or self._tag_stack[-1] not in {"script", "style", "title"}:
            self._text_chunks.append(text)

    def to_payload(self) -> dict[str, object]:
        return {
            "title": " ".join(self._title_chunks).strip(),
            "text": " ".join(self._text_chunks),
            "links": self._links,
            "forms": self._forms,
        }


def _label(attrs: dict[str, str]) -> str:
    return attrs.get("aria-label") or attrs.get("placeholder") or attrs.get("name") or attrs.get("id") or ""


def _preview_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:MAX_TEXT_PREVIEW]
