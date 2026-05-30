from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BrowserStepDecision(BaseModel):
    action_type: Literal["CALL_TOOL", "ASK_USER", "ANSWER", "STOP", "BLOCK"]
    decision_summary: str
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    expected_observation: str | None = None
    risk_notes: list[str] = Field(default_factory=list)


class BrowserTaskBudget(BaseModel):
    max_steps: int = 5
    max_tool_calls: int = 5
    max_seconds: float = 30.0


BROWSER_TASK_EVENT_TYPES = (
    "browser_task_started",
    "browser_observation_added",
    "browser_step_decided",
    "browser_action_blocked",
    "browser_task_completed",
    "browser_task_failed",
)
