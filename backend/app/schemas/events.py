from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AssistantEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    run_id: str | None = None
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @staticmethod
    def parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value)
