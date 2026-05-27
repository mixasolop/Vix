from typing import Protocol

from app.assistant.planner import FALLBACK_MESSAGE
from app.schemas.plans import AssistantPlan, PlanStep, PlanStepStatus
from app.schemas.tools import ToolDefinition


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, object]]) -> str:
        """Return a normal assistant reply. Stage 1 does not use this to execute tools."""

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        """Return a validated plan shape for future LLM tool planning."""


class DeterministicLLMClient:
    async def complete(self, messages: list[dict[str, object]]) -> str:
        return FALLBACK_MESSAGE

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        return AssistantPlan(
            goal=f"Respond to: {user_text.strip()}" if user_text.strip() else "Respond to empty request",
            steps=[
                PlanStep(number=1, title="Produce normal assistant reply", status=PlanStepStatus.pending),
            ],
        )


class OpenAILLMClient:
    async def complete(self, messages: list[dict[str, object]]) -> str:
        raise NotImplementedError("OpenAI integration starts after the Stage 1 deterministic runtime is stable.")

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        raise NotImplementedError("OpenAI planning starts after deterministic planner and policy checks are stable.")
