from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    app_name: str = "Desktop Assistant"
    environment: str = "local"
    database_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "assistant_events.sqlite3"
    )


@lru_cache
def load_config() -> AppConfig:
    return AppConfig()
