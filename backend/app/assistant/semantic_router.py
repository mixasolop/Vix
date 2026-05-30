from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.assistant.context_intent import ContextIntent, ContextOperation, ContextSource
from app.assistant.policy import PolicyDecision, PolicyEngine
from app.assistant.planner import ToolProposal
from app.assistant.request_classifier import RequestCategory, RequestClassification
from app.schemas.tools import ToolStatus
from app.tools.registry import ToolRegistry

LOGGER = logging.getLogger("app.assistant.semantic_router")

SEMANTIC_ROUTER_SYSTEM_PROMPT = """
You are Vix's semantic intent router.
Return JSON only. Do not include Markdown, prose, comments, or code fences.
You classify intent only; you never execute tools.
If confidence is below 0.75, return ASK_CLARIFICATION.
Do not include chain-of-thought. Use reason_summary only.
Bare app aliases such as "calc", "calculator", or "notepad" are ASK_CLARIFICATION, not LOCAL_TOOL.
Negated actions are NO_ACTION. How-to and hypothetical questions are GENERAL_ANSWER.
""".strip()

SEMANTIC_ROUTER_MIN_CONFIDENCE = 0.75


class SemanticRouterLLMClient(Protocol):
    async def complete(self, messages: list[dict[str, object]]) -> str:
        """Return the raw semantic-route JSON string."""


class SemanticRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal[
        "GENERAL_ANSWER",
        "LOCAL_TOOL",
        "REALTIME_INFO",
        "LOCAL_CONTEXT",
        "MISSING_TOOL",
        "NO_ACTION",
        "ASK_CLARIFICATION",
        "UNSAFE_OR_BLOCKED",
    ]
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_summary: str = Field(min_length=1)
    context_operation: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_route_shape(self) -> SemanticRoute:
        if self.needs_clarification and self.category != "ASK_CLARIFICATION":
            raise ValueError("needs_clarification may only be true for ASK_CLARIFICATION.")
        if self.category == "ASK_CLARIFICATION" and self.needs_clarification and not self.clarification_question:
            raise ValueError("ASK_CLARIFICATION with needs_clarification requires clarification_question.")
        if self.category not in {"LOCAL_TOOL", "REALTIME_INFO", "LOCAL_CONTEXT"} and self.tool_name is not None:
            raise ValueError("Only executable tool categories may include tool_name.")
        if self.tool_name is None and self.arguments:
            raise ValueError("arguments require tool_name.")
        return self


class SemanticRouteValidationError(ValueError):
    """Raised when a semantic-router result is invalid or unsafe to route."""


@dataclass(frozen=True)
class ValidatedSemanticRoute:
    route: SemanticRoute
    classification: RequestClassification
    policy_decision: PolicyDecision | None = None


class SemanticRouter:
    def __init__(self, *, llm_client: SemanticRouterLLMClient, registry: ToolRegistry, policy_engine: PolicyEngine) -> None:
        self._llm_client = llm_client
        self._registry = registry
        self._policy_engine = policy_engine

    async def route(self, *, original_message: str, canonical_message: str) -> ValidatedSemanticRoute:
        raw_output = await self._llm_client.complete(
            [
                {"role": "system", "content": SEMANTIC_ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_message": original_message,
                            "canonical_message": canonical_message,
                            "implemented_tools": [
                                _tool_payload(tool)
                                for tool in self._registry.list_tools()
                                if tool.status == ToolStatus.implemented
                            ],
                            "planned_tools": [
                                _tool_payload(tool)
                                for tool in self._registry.list_tools()
                                if tool.status == ToolStatus.planned
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
        )
        try:
            route = SemanticRoute.model_validate_json(raw_output)
        except (ValidationError, ValueError) as exc:
            raise SemanticRouteValidationError(f"Invalid semantic route JSON: {exc}") from exc

        return validate_semantic_route(
            route,
            registry=self._registry,
            policy_engine=self._policy_engine,
        )


def validate_semantic_route(
    route: SemanticRoute,
    *,
    registry: ToolRegistry,
    policy_engine: PolicyEngine,
) -> ValidatedSemanticRoute:
    if route.confidence < SEMANTIC_ROUTER_MIN_CONFIDENCE:
        route = SemanticRoute(
            category="ASK_CLARIFICATION",
            tool_name=None,
            arguments={},
            confidence=route.confidence,
            reason_summary=f"Semantic router confidence below {SEMANTIC_ROUTER_MIN_CONFIDENCE}: {route.reason_summary}",
            needs_clarification=True,
            clarification_question=route.clarification_question or "Can you clarify what you want me to do?",
        )

    policy_decision: PolicyDecision | None = None
    proposal: ToolProposal | None = None
    if route.tool_name is not None:
        tool = registry.get(route.tool_name)
        if tool is None:
            raise SemanticRouteValidationError(f"Semantic router selected unregistered tool: {route.tool_name}.")
        if tool.status != ToolStatus.implemented:
            raise SemanticRouteValidationError(f"Semantic router selected non-implemented tool: {route.tool_name}.")

        validation_error = registry.validate_arguments(route.tool_name, route.arguments)
        if validation_error is not None:
            raise SemanticRouteValidationError(f"Semantic router produced invalid arguments for {route.tool_name}: {validation_error}")

        policy_decision = policy_engine.evaluate_tool_call(tool, route.arguments)
        if policy_decision.blocked:
            raise SemanticRouteValidationError(f"Semantic router route blocked by policy: {policy_decision.reason}")
        proposal = ToolProposal(route.tool_name, dict(route.arguments))

    context_intent = _semantic_context_intent(route, proposal)
    classification = RequestClassification(
        category=RequestCategory(route.category),
        reason=route.reason_summary,
        tool_proposal=proposal,
        missing_input="clarification" if route.category == "ASK_CLARIFICATION" else None,
        context_intent=context_intent,
        router_source="semantic",
    )
    return ValidatedSemanticRoute(route=route, classification=classification, policy_decision=policy_decision)


def _tool_payload(tool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "status": tool.status.value,
        "risk_level": tool.risk_level.value,
        "confirmation_policy": tool.confirmation_policy.value,
        "input_schema": tool.input_schema,
    }


def _semantic_context_intent(route: SemanticRoute, proposal: ToolProposal | None) -> ContextIntent | None:
    if route.category != "LOCAL_CONTEXT" or proposal is None:
        return None

    sources = {
        "get_selected_text": ContextSource.selected_text,
        "get_clipboard_text": ContextSource.clipboard,
        "get_context_window_info": ContextSource.context_window,
        "get_foreground_window_info": ContextSource.foreground_window,
    }
    source = sources.get(proposal.name)
    if source is None:
        return None

    try:
        operation = ContextOperation(route.context_operation or "unknown")
    except ValueError:
        operation = ContextOperation.unknown

    return ContextIntent(
        source=source,
        operation=operation,
        tool_name=proposal.name,
        arguments=dict(proposal.arguments),
        reason=route.reason_summary,
        confidence=route.confidence,
    )
