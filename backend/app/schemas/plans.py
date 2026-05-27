from enum import Enum

from pydantic import BaseModel, Field


class PlanStepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"


class PlanStep(BaseModel):
    number: int = Field(ge=1)
    title: str
    status: PlanStepStatus = PlanStepStatus.pending


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
