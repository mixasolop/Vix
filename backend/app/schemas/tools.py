from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ConfirmationPolicy(str, Enum):
    none = "none"
    before_execute = "before_execute"


class ToolStatus(str, Enum):
    implemented = "implemented"
    planned = "planned"
    disabled = "disabled"


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0)


class ToolDefinition(BaseModel):
    name: str
    description: str
    status: ToolStatus = ToolStatus.planned
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_level: RiskLevel
    confirmation_policy: ConfirmationPolicy
    timeout_seconds: int = Field(gt=0)
    retry_policy: RetryPolicy


class ToolResult(BaseModel):
    tool: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any]
    status: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PermissionRequest(BaseModel):
    permission_id: str
    tool: str
    reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PermissionDecisionResponse(BaseModel):
    permission_id: str
    status: str


class ToolListResponse(BaseModel):
    tools: list[ToolDefinition]
