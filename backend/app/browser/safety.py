from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse

from app.schemas.tools import RiskLevel


HIGH_RISK_TERMS = (
    "submit",
    "book",
    "reserve",
    "reservation",
    "order",
    "pay",
    "payment",
    "checkout",
    "confirm",
    "send",
    "login",
    "log in",
    "password",
    "card",
    "credit card",
    "address",
    "phone",
    "email",
    "upload",
    "download",
)

BLOCKED_TERMS = (
    "captcha",
    "password",
    "credit card",
    "card number",
    "cvv",
    "checkout",
    "payment",
    "download exe",
    "download executable",
    "upload file",
)


@dataclass(frozen=True)
class BrowserSafetyAssessment:
    risk_level: RiskLevel
    blocked: bool
    reason: str


def assess_browser_text(text: str) -> BrowserSafetyAssessment:
    normalized = _normalize(text)
    if any(term in normalized for term in BLOCKED_TERMS):
        return BrowserSafetyAssessment(
            risk_level=RiskLevel.high_risk,
            blocked=True,
            reason="Browser action is blocked in Stage 5 because it appears to involve credentials, payment, CAPTCHA, upload, or executable download.",
        )
    if any(term in normalized for term in HIGH_RISK_TERMS):
        return BrowserSafetyAssessment(
            risk_level=RiskLevel.high_risk,
            blocked=False,
            reason="Browser action contains high-risk terms and requires explicit permission.",
        )
    return BrowserSafetyAssessment(
        risk_level=RiskLevel.low_write,
        blocked=False,
        reason="Browser action appears to be normal navigation or reading.",
    )


def risk_hint_for_text(text: str | None, href: str | None = None) -> str | None:
    combined = " ".join(part for part in (text or "", href or "") if part)
    assessment = assess_browser_text(combined)
    if assessment.blocked:
        return "blocked"
    if assessment.risk_level == RiskLevel.high_risk:
        return "high_risk"
    return None


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise ValueError("URL is required.")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        return value
    if re.match(r"^[\w.-]+\.[a-z]{2,}(/.*)?$", value, flags=re.IGNORECASE):
        return f"https://{value}"
    raise ValueError("Only http, https, and file URLs are supported in Stage 5.")


def is_blocked_browser_request(text: str) -> bool:
    normalized = _normalize(text)
    return any(term in normalized for term in BLOCKED_TERMS)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
