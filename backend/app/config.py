from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = BACKEND_DIR / ".env"
DEFAULT_DATABASE_PATH = BACKEND_DIR / "assistant_events.sqlite3"


class AppConfig(BaseModel):
    app_name: str = "Desktop Assistant"
    environment: str = "local"
    database_path: Path = Field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    config_file_path: Path = Field(default_factory=lambda: DEFAULT_CONFIG_FILE)
    openai_api_key: str | None = None
    ai_provider: str = "openai"
    ai_proposal_model: str = "gpt-5.4-mini"
    ai_general_answers_enabled: bool = True
    ai_proposals_enabled: bool = False


@lru_cache
def load_config() -> AppConfig:
    return load_config_from_file(DEFAULT_CONFIG_FILE)


def load_config_from_file(config_file_path: Path = DEFAULT_CONFIG_FILE) -> AppConfig:
    file_values = _read_key_value_file(config_file_path)
    return AppConfig(
        config_file_path=config_file_path,
        openai_api_key=_optional_config_value("OPENAI_API_KEY", file_values),
        ai_provider=_config_value("AI_PROVIDER", file_values, "openai"),
        ai_proposal_model=_config_value("AI_PROPOSAL_MODEL", file_values, "gpt-5.4-mini"),
        ai_general_answers_enabled=_config_bool("AI_GENERAL_ANSWERS_ENABLED", file_values, default=True),
        ai_proposals_enabled=_config_bool("AI_PROPOSALS_ENABLED", file_values, default=False),
    )


def _read_key_value_file(config_file_path: Path) -> dict[str, str]:
    if not config_file_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in config_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_optional_quotes(value.strip())
    return values


def _config_value(name: str, file_values: dict[str, str], default: str) -> str:
    value = file_values.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _optional_config_value(name: str, file_values: dict[str, str]) -> str | None:
    value = file_values.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _config_bool(name: str, file_values: dict[str, str], default: bool) -> bool:
    value = file_values.get(name)
    return _str_bool(value, default)


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _str_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
