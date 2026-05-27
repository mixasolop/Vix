from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    app_name: str = "Desktop Assistant"
    environment: str = "local"
    database_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "assistant_events.sqlite3"
    )
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    ai_provider: str = Field(default_factory=lambda: os.getenv("AI_PROVIDER", "openai"))
    ai_proposal_model: str = Field(default_factory=lambda: os.getenv("AI_PROPOSAL_MODEL", "gpt-5.4-mini"))
    ai_proposals_enabled: bool = Field(default_factory=lambda: _env_bool("AI_PROPOSALS_ENABLED", default=False))


@lru_cache
def load_config() -> AppConfig:
    return AppConfig()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
