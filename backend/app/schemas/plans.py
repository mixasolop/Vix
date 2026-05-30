from enum import Enum
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class PlanStepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class PlanStep(BaseModel):
    number: int = Field(ge=1)
    key: str | None = None
    title: str
    status: PlanStepStatus = PlanStepStatus.pending
    tool_name: str | None = None
    expected_observation: str | None = None
    risk_level: str | None = None
    requires_permission: bool | None = None


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]


class AssistantPlan(Plan):
    """Validated plan shape for future LLM planner output."""


def update_plan_step(
    plan: Plan,
    step_number: int,
    status: PlanStepStatus | None = None,
    **updates: Any,
) -> Plan:
    """Return a copy of a plan with one step updated, preserving step metadata."""
    updated_steps: list[PlanStep] = []
    for step in plan.steps:
        if step.number != step_number:
            updated_steps.append(step)
            continue

        values = dict(updates)
        if status is not None:
            values["status"] = status
        updated_steps.append(step.model_copy(update=values))

    return Plan(goal=plan.goal, steps=updated_steps)


def update_plan_steps(plan: Plan, statuses: Mapping[int, PlanStepStatus]) -> Plan:
    """Return a copy of a plan with arbitrary step statuses updated."""
    updated_steps = [
        step.model_copy(update={"status": statuses[step.number]}) if step.number in statuses else step
        for step in plan.steps
    ]
    return Plan(goal=plan.goal, steps=updated_steps)
