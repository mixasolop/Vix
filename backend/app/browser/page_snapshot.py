from __future__ import annotations

from app.schemas.browser import BrowserPageSnapshot


def snapshot_summary(snapshot: BrowserPageSnapshot) -> dict[str, object]:
    return {
        "url": snapshot.url,
        "title": snapshot.title,
        "text_preview": snapshot.text_preview,
        "text_length": snapshot.text_length,
        "links_count": len(snapshot.links),
        "forms_count": len(snapshot.forms),
        "captured_at": snapshot.captured_at.isoformat(),
    }


def snapshot_message(snapshot: BrowserPageSnapshot) -> str:
    title = snapshot.title or snapshot.url
    return f"Read page '{title}' with {snapshot.text_length} characters, {len(snapshot.links)} links, and {len(snapshot.forms)} forms."
