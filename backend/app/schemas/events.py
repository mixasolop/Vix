from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AssistantEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @staticmethod
    def parse_created_at(value: str) -> datetime:
        return datetime.fromisoformat(value)
