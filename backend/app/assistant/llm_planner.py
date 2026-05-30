from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.assistant.policy import PolicyEngine
from app.schemas.tools import RiskLevel, ToolDefinition
from app.tools.registry import ToolRegistry


LLM_PLANNER_SYSTEM_PROMPT = """
You are Vix's future planning module for complex desktop-assistant tasks.

Return JSON only. Do not include Markdown, prose, comments, or code fences.
The JSON must match the LLMPlan schema exactly.
Do not include chain-of-thought or private reasoning. Use reason_summary only.
Propose one safe next step at a time. After one action, the app will observe and replan.
Never claim a tool was executed. The application validates, authorizes, and executes tools separately.
""".strip()


class PlannerLLMClient(Protocol):
    async def complete(self, messages: list[dict[str, object]]) -> str:
        """Return the raw planner JSON string."""


class PlannedAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["call_tool", "ask_user", "answer", "stop"]
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    user_question: str | None = None
    reason_summary: str = Field(min_length=1)
    risk_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_shape(self) -> PlannedAction:
        if self.action_type == "call_tool" and not self.tool_name:
            raise ValueError("call_tool actions require tool_name.")
        if self.action_type != "call_tool" and self.tool_name is not None:
            raise ValueError("Only call_tool actions may include tool_name.")
        if self.action_type == "ask_user" and not self.user_question:
            raise ValueError("ask_user actions require user_question.")
        return self


class LLMPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    actions: list[PlannedAction] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


class LLMPlanValidationError(ValueError):
    """Raised when a model-produced plan is not safe or valid enough to consider."""


@dataclass(frozen=True)
class NextActionPlan:
    plan: LLMPlan
    action: PlannedAction | None


class LLMPlanner:
    """Validate future LLM-generated plans without executing them.

    This interface is intentionally separate from deterministic routing. It prepares
    the future loop shape: ask for a plan, validate it, take one step, observe, replan.
    """

    def __init__(self, *, llm_client: PlannerLLMClient, registry: ToolRegistry, policy_engine: PolicyEngine) -> None:
        self._llm_client = llm_client
        self._registry = registry
        self._policy_engine = policy_engine

    async def create_plan(
        self,
        *,
        user_goal: str,
        observations: list[dict[str, object] | str] | None = None,
    ) -> LLMPlan:
        raw_output = await self._llm_client.complete(
            [
                {"role": "system", "content": LLM_PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_goal": user_goal,
                            "observations": observations or [],
                            "available_tools": [_tool_payload(tool) for tool in self._registry.list_tools()],
                            "planning_contract": {
                                "json_only": True,
                                "execute_one_step_then_replan": True,
                                "no_chain_of_thought": True,
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
        )

        try:
            plan = LLMPlan.model_validate_json(raw_output)
        except (ValidationError, ValueError) as exc:
            raise LLMPlanValidationError(f"Invalid LLM plan JSON: {exc}") from exc

        self.validate_plan(plan)
        return plan

    async def create_next_action_plan(
        self,
        *,
        user_goal: str,
        observations: list[dict[str, object] | str] | None = None,
    ) -> NextActionPlan:
        plan = await self.create_plan(user_goal=user_goal, observations=observations)
        return NextActionPlan(plan=plan, action=self.next_action(plan))

    def next_action(self, plan: LLMPlan) -> PlannedAction | None:
        """Return only the next step. The caller must observe and replan after it."""

        return plan.actions[0] if plan.actions else None

    def validate_plan(self, plan: LLMPlan) -> None:
        for index, action in enumerate(plan.actions, start=1):
            if action.action_type == "call_tool":
                self._validate_tool_action(index, action)

    def _validate_tool_action(self, index: int, action: PlannedAction) -> None:
        assert action.tool_name is not None

        tool = self._registry.get(action.tool_name)
        if tool is None:
            raise LLMPlanValidationError(f"Action {index} references unknown tool: {action.tool_name}.")

        validation_error = self._registry.validate_arguments(action.tool_name, action.arguments)
        if validation_error is not None:
            raise LLMPlanValidationError(f"Action {index} has invalid arguments for {action.tool_name}: {validation_error}")

        policy_decision = self._policy_engine.evaluate_tool_call(tool, action.arguments)
        if policy_decision.blocked:
            raise LLMPlanValidationError(f"Action {index} is blocked by policy: {policy_decision.reason}")

        if _is_write_or_high_risk(tool) and not policy_decision.requires_permission:
            raise LLMPlanValidationError(
                f"Action {index} uses {tool.risk_level.value} tool {tool.name}, but policy did not require permission."
            )

        if not policy_decision.allowed and not policy_decision.requires_permission:
            raise LLMPlanValidationError(f"Action {index} is not allowed by policy: {policy_decision.reason}")


def _is_write_or_high_risk(tool: ToolDefinition) -> bool:
    return tool.risk_level in {RiskLevel.low_write, RiskLevel.medium_write, RiskLevel.high_risk}


def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "status": tool.status.value,
        "risk_level": tool.risk_level.value,
        "confirmation_policy": tool.confirmation_policy.value,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
    }
