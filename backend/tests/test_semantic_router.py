import asyncio
import json

import pytest

from app.assistant.policy import PolicyEngine
from app.assistant.semantic_router import SemanticRoute, SemanticRouteValidationError, SemanticRouter, validate_semantic_route
from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition, ToolStatus
from app.tools.registry import ToolRegistry


class FakeLLMClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: list[dict[str, object]] = []

    async def complete(self, messages: list[dict[str, object]]) -> str:
        self.messages = messages
        return self.output


def test_semantic_router_validates_fake_llm_route_to_implemented_tool() -> None:
    registry = ToolRegistry()
    registry.register(_tool_definition("get_clipboard_text"))
    llm = FakeLLMClient(
        json.dumps(
            {
                "category": "LOCAL_CONTEXT",
                "tool_name": "get_clipboard_text",
                "arguments": {},
                "confidence": 0.92,
                "reason_summary": "User asked what is in the clipboard.",
                "needs_clarification": False,
                "clarification_question": None,
            }
        )
    )
    router = SemanticRouter(llm_client=llm, registry=registry, policy_engine=PolicyEngine())

    result = asyncio.run(router.route(original_message="what word i have in clip board", canonical_message="what word i have in clipboard"))

    assert result.classification.category.value == "LOCAL_CONTEXT"
    assert result.classification.tool_proposal is not None
    assert result.classification.tool_proposal.name == "get_clipboard_text"
    assert "implemented_tools" in str(llm.messages[1]["content"])


def test_semantic_router_rejects_unregistered_tool() -> None:
    registry = ToolRegistry()
    route = SemanticRoute(
        category="LOCAL_CONTEXT",
        tool_name="ghost_tool",
        arguments={},
        confidence=0.95,
        reason_summary="Invalid tool should be rejected.",
    )

    with pytest.raises(SemanticRouteValidationError, match="unregistered tool"):
        validate_semantic_route(route, registry=registry, policy_engine=PolicyEngine())


def test_semantic_router_rejects_invalid_tool_arguments() -> None:
    registry = ToolRegistry()
    registry.register(_tool_definition("get_weather", properties={"location": {"type": "string"}}, required=["location"]))
    route = SemanticRoute(
        category="REALTIME_INFO",
        tool_name="get_weather",
        arguments={},
        confidence=0.95,
        reason_summary="Missing required weather location.",
    )

    with pytest.raises(SemanticRouteValidationError, match="Missing required argument"):
        validate_semantic_route(route, registry=registry, policy_engine=PolicyEngine())


def test_semantic_router_low_confidence_becomes_ask_clarification() -> None:
    registry = ToolRegistry()
    route = SemanticRoute(
        category="GENERAL_ANSWER",
        tool_name=None,
        arguments={},
        confidence=0.4,
        reason_summary="Unclear user intent.",
    )

    result = validate_semantic_route(route, registry=registry, policy_engine=PolicyEngine())

    assert result.route.category == "ASK_CLARIFICATION"
    assert result.classification.category.value == "ASK_CLARIFICATION"
    assert result.classification.tool_proposal is None
    assert result.route.needs_clarification is True


def test_semantic_router_high_risk_tool_still_requires_permission() -> None:
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
    route = SemanticRoute(
        category="LOCAL_TOOL",
        tool_name="send_message",
        arguments={"recipient": "Anna", "message": "I will be late."},
        confidence=0.9,
        reason_summary="User asked to send an external message.",
    )

    result = validate_semantic_route(route, registry=registry, policy_engine=PolicyEngine())

    assert result.policy_decision is not None
    assert result.policy_decision.requires_permission is True
    assert result.classification.tool_proposal is not None
    assert result.classification.tool_proposal.name == "send_message"


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
        description=f"Fake semantic-router test tool: {name}",
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
