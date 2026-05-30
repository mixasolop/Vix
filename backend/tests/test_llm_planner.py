import asyncio
import json

import pytest

from app.assistant.llm_planner import LLMPlanValidationError, LLMPlanner
from app.assistant.policy import PolicyEngine
from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition, ToolResult, ToolStatus
from app.tools.registry import ToolRegistry


class FakeLLMClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: list[dict[str, object]] = []

    async def complete(self, messages: list[dict[str, object]]) -> str:
        self.messages = messages
        return self.output


def test_llm_planner_accepts_valid_json_plan_and_returns_next_action() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool_definition(
            "search_notes",
            properties={"query": {"type": "string"}},
            required=["query"],
        )
    )
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "search_notes",
                    "arguments": {"query": "kafka"},
                    "reason_summary": "Search notes for the requested topic.",
                }
            ]
        )
    )

    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    result = asyncio.run(planner.create_next_action_plan(user_goal="find my Kafka notes"))

    assert result.plan.goal == "find my Kafka notes"
    assert result.action is not None
    assert result.action.tool_name == "search_notes"
    assert result.action.arguments == {"query": "kafka"}
    assert "Return JSON only" in str(llm.messages[0]["content"])
    assert "no_chain_of_thought" in str(llm.messages[1]["content"])


def test_llm_planner_rejects_non_json_output() -> None:
    registry = ToolRegistry()
    llm = FakeLLMClient("Here is the plan: {}")
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="Invalid LLM plan JSON"):
        asyncio.run(planner.create_plan(user_goal="do something complex"))


def test_llm_planner_rejects_unknown_tool_name() -> None:
    registry = ToolRegistry()
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "unknown_tool",
                    "arguments": {},
                    "reason_summary": "Try a tool that does not exist.",
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="unknown tool"):
        asyncio.run(planner.create_plan(user_goal="use a nonexistent tool"))


def test_llm_planner_rejects_invalid_tool_arguments() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool_definition(
            "search_notes",
            properties={"query": {"type": "string"}},
            required=["query"],
        )
    )
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "search_notes",
                    "arguments": {},
                    "reason_summary": "Search without providing the required query.",
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="Missing required argument"):
        asyncio.run(planner.create_plan(user_goal="find my Kafka notes"))


def test_llm_planner_accepts_high_risk_action_only_as_permission_required_plan() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool_definition(
            "send_message",
            properties={"recipient": {"type": "string"}, "message": {"type": "string"}},
            required=["recipient", "message"],
            risk_level=RiskLevel.high_risk,
            confirmation_policy=ConfirmationPolicy.before_execute,
        )
    )
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "send_message",
                    "arguments": {"recipient": "Anna", "message": "I will be late."},
                    "reason_summary": "Drafting an external message requires human permission before execution.",
                    "risk_notes": ["External communication must be approved."],
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    plan = asyncio.run(planner.create_plan(user_goal="tell Anna I will be late"))

    assert plan.actions[0].tool_name == "send_message"


def test_llm_planner_rejects_write_action_when_policy_does_not_require_permission() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool_definition(
            "launch_app",
            properties={"app_name": {"type": "string"}},
            required=["app_name"],
            risk_level=RiskLevel.low_write,
            confirmation_policy=ConfirmationPolicy.none,
        )
    )
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "launch_app",
                    "arguments": {"app_name": "calculator"},
                    "reason_summary": "Launching an app is a write-like local action.",
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="policy did not require permission"):
        asyncio.run(planner.create_plan(user_goal="open calculator"))


def test_llm_planner_rejects_chain_of_thought_extra_fields() -> None:
    registry = ToolRegistry()
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "answer",
                    "tool_name": None,
                    "arguments": {},
                    "reason_summary": "Answer directly.",
                    "chain_of_thought": "private hidden reasoning should never be accepted",
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="Invalid LLM plan JSON"):
        asyncio.run(planner.create_plan(user_goal="explain recursion"))


def test_llm_planner_ask_user_action_requires_user_question() -> None:
    registry = ToolRegistry()
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "ask_user",
                    "tool_name": None,
                    "arguments": {},
                    "reason_summary": "Need a location before checking weather.",
                }
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    with pytest.raises(LLMPlanValidationError, match="Invalid LLM plan JSON"):
        asyncio.run(planner.create_plan(user_goal="check weather"))


def test_llm_planner_selects_one_step_without_executing_tool() -> None:
    registry = ToolRegistry()
    executed = False

    async def exploding_tool(arguments: dict[str, object]) -> ToolResult:
        nonlocal executed
        executed = True
        raise AssertionError("LLMPlanner must not execute tools.")

    registry.register(_tool_definition("read_context"), exploding_tool)
    llm = FakeLLMClient(
        _plan_json(
            [
                {
                    "action_type": "call_tool",
                    "tool_name": "read_context",
                    "arguments": {},
                    "reason_summary": "Read context as the next single step.",
                },
                {
                    "action_type": "answer",
                    "tool_name": None,
                    "arguments": {},
                    "reason_summary": "Answer only after observing context.",
                },
            ]
        )
    )
    planner = LLMPlanner(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    result = asyncio.run(planner.create_next_action_plan(user_goal="explain selected text"))

    assert result.action is not None
    assert result.action.tool_name == "read_context"
    assert executed is False


def _tool_definition(
    name: str,
    *,
    properties: dict[str, object] | None = None,
    required: list[str] | None = None,
    risk_level: RiskLevel = RiskLevel.read,
    confirmation_policy: ConfirmationPolicy = ConfirmationPolicy.none,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Fake planner test tool: {name}",
        status=ToolStatus.implemented,
        input_schema={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        risk_level=risk_level,
        confirmation_policy=confirmation_policy,
        timeout_seconds=5,
        retry_policy=RetryPolicy(max_attempts=1),
    )


def _plan_json(actions: list[dict[str, object]], *, goal: str = "find my Kafka notes") -> str:
    return json.dumps(
        {
            "goal": goal,
            "assumptions": [],
            "missing_information": [],
            "actions": actions,
            "stop_conditions": ["Stop after one action and replan from observation."],
        }
    )
