from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_PROPOSED_TOOL_RISK_LEVELS = {"READ", "LOW_WRITE", "MEDIUM_WRITE", "HIGH_RISK"}


class ProposedToolStatus(StrEnum):
    proposed = "proposed"
    approved = "approved"
    rejected = "rejected"
    needs_changes = "needs_changes"


class ProposedToolDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    reason: str
    risk_level: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value: str) -> str:
        if value not in ALLOWED_PROPOSED_TOOL_RISK_LEVELS:
            raise ValueError(f"risk_level must be one of: {', '.join(sorted(ALLOWED_PROPOSED_TOOL_RISK_LEVELS))}")
        return value


class ProposedTool(ProposedToolDraft):
    id: str
    status: ProposedToolStatus
    created_from_message: str
    created_at: str
    updated_at: str


class CreateProposedToolRequest(ProposedToolDraft):
    created_from_message: str


class ProposedToolListResponse(BaseModel):
    tools: list[ProposedTool]
