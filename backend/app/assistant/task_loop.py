from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import inspect
from time import monotonic
from typing import Literal

from app.assistant.policy import PolicyEngine
from app.schemas.tools import ToolResult
from app.tools.registry import ToolRegistry


TaskStatus = Literal["running", "waiting_for_user", "completed", "failed", "blocked"]
ActionType = Literal["CALL_TOOL", "ASK_USER", "ANSWER", "STOP", "BLOCK"]


@dataclass
class Observation:
    source: str
    content: dict[str, object] | str
    created_at: datetime


@dataclass
class ExecutedStep:
    step_number: int
    tool_name: str
    arguments: dict[str, object]
    status: str
    observation: Observation
    decision_summary: str
    error: str | None = None


@dataclass
class TaskState:
    session_id: str
    run_id: str
    user_goal: str
    observations: list[Observation] = field(default_factory=list)
    steps_taken: list[ExecutedStep] = field(default_factory=list)
    status: TaskStatus = "running"


@dataclass
class NextActionDecision:
    action_type: ActionType
    decision_summary: str
    tool_name: str | None
    arguments: dict[str, object] = field(default_factory=dict)
    missing_input: str | None = None
    risk_level: str | None = None


@dataclass(frozen=True)
class TaskLoopBudget:
    max_steps: int = 5
    max_tool_calls: int = 5
    max_seconds: float = 15.0


DecisionProvider = Callable[[TaskState], NextActionDecision | Awaitable[NextActionDecision]]
TaskEventEmitter = Callable[[str, str, str, dict[str, object]], Awaitable[None]]


class TaskLoop:
    """Small ReAct-style loop that exposes summaries, observations, and actions only."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy_engine: PolicyEngine,
        decide_next_action: DecisionProvider,
        emit_event: TaskEventEmitter,
        budget: TaskLoopBudget | None = None,
    ) -> None:
        self._registry = registry
        self._policy_engine = policy_engine
        self._decide_next_action = decide_next_action
        self._emit_event = emit_event
        self._budget = budget or TaskLoopBudget()

    async def run(self, *, session_id: str, run_id: str, user_goal: str) -> TaskState:
        state = TaskState(session_id=session_id, run_id=run_id, user_goal=user_goal)
        started_at = monotonic()
        await self._emit("task_loop_started", state, {"budget": _budget_payload(self._budget)})

        for step_index in range(1, self._budget.max_steps + 1):
            if monotonic() - started_at > self._budget.max_seconds:
                return await self._fail(state, "max_seconds", "Task loop exceeded its time budget.")

            decision = await self._next_decision(state)
            await self._emit(
                "task_step_decided",
                state,
                {
                    "step_index": step_index,
                    "decision": _decision_payload(decision),
                },
            )

            if decision.action_type == "ASK_USER":
                state.status = "waiting_for_user"
                await self._emit_completed(state, "waiting_for_user", decision)
                return state

            if decision.action_type == "ANSWER":
                state.status = "completed"
                await self._emit_completed(state, "answered", decision)
                return state

            if decision.action_type == "STOP":
                state.status = "completed"
                await self._emit_completed(state, "stopped", decision)
                return state

            if decision.action_type == "BLOCK":
                state.status = "blocked"
                return await self._fail(state, "blocked_by_decision", decision.decision_summary, decision=decision)

            if decision.action_type == "CALL_TOOL":
                if len(state.steps_taken) >= self._budget.max_tool_calls:
                    return await self._fail(state, "max_tool_calls", "Task loop exceeded its tool-call budget.", decision=decision)

                blocked_state = await self._block_if_tool_not_allowed(state, decision)
                if blocked_state is not None:
                    return blocked_state

                await self._call_tool(state, step_index, decision)

        return await self._fail(state, "max_steps", "Task loop reached the maximum number of steps.")

    async def _next_decision(self, state: TaskState) -> NextActionDecision:
        decision = self._decide_next_action(state)
        if inspect.isawaitable(decision):
            decision = await decision
        return decision

    async def _block_if_tool_not_allowed(self, state: TaskState, decision: NextActionDecision) -> TaskState | None:
        if not decision.tool_name:
            return await self._fail(state, "missing_tool_name", "CALL_TOOL decision did not include a tool name.", decision=decision)

        tool = self._registry.get(decision.tool_name)
        if tool is None:
            return await self._fail(state, "unknown_tool", f"Unknown tool: {decision.tool_name}.", decision=decision)

        policy_decision = self._policy_engine.evaluate_tool_call(tool, decision.arguments)
        if policy_decision.blocked or policy_decision.requires_permission or not policy_decision.allowed:
            state.status = "blocked"
            await self._emit(
                "task_loop_failed",
                state,
                {
                    "status": state.status,
                    "stop_reason": "blocked_by_policy",
                    "message": policy_decision.reason,
                    "decision": _decision_payload(decision),
                    "policy_decision": {
                        "allowed": policy_decision.allowed,
                        "requires_permission": policy_decision.requires_permission,
                        "blocked": policy_decision.blocked,
                        "reason": policy_decision.reason,
                    },
                    "state": _state_payload(state),
                },
            )
            return state

        return None

    async def _call_tool(self, state: TaskState, step_index: int, decision: NextActionDecision) -> None:
        assert decision.tool_name is not None
        result = await self._registry.execute(decision.tool_name, decision.arguments)
        observation = _observation_from_tool_result(result)
        state.observations.append(observation)
        state.steps_taken.append(
            ExecutedStep(
                step_number=step_index,
                tool_name=decision.tool_name,
                arguments=dict(decision.arguments),
                status=result.status,
                observation=observation,
                decision_summary=decision.decision_summary,
                error=result.error,
            )
        )
        await self._emit(
            "task_observation_added",
            state,
            {
                "step_index": step_index,
                "observation": _observation_payload(observation),
                "tool_result": {
                    "tool": result.tool,
                    "status": result.status,
                    "output": result.output,
                    "error": result.error,
                },
                "state": _state_payload(state),
            },
        )

    async def _emit_completed(self, state: TaskState, stop_reason: str, decision: NextActionDecision) -> None:
        await self._emit(
            "task_loop_completed",
            state,
            {
                "status": state.status,
                "stop_reason": stop_reason,
                "decision_summary": decision.decision_summary,
                "missing_input": decision.missing_input,
                "answer": decision.arguments.get("answer"),
                "state": _state_payload(state),
            },
        )

    async def _fail(
        self,
        state: TaskState,
        stop_reason: str,
        message: str,
        decision: NextActionDecision | None = None,
    ) -> TaskState:
        if state.status == "running":
            state.status = "failed"
        await self._emit(
            "task_loop_failed",
            state,
            {
                "status": state.status,
                "stop_reason": stop_reason,
                "message": message,
                "decision": _decision_payload(decision) if decision is not None else None,
                "state": _state_payload(state),
            },
        )
        return state

    async def _emit(self, event_type: str, state: TaskState, data: dict[str, object]) -> None:
        await self._emit_event(event_type, state.session_id, state.run_id, data)


def _observation_from_tool_result(result: ToolResult) -> Observation:
    content: dict[str, object] = {
        "status": result.status,
        "output": result.output,
        "error": result.error,
    }
    return Observation(source=result.tool, content=content, created_at=datetime.now(UTC))


def _budget_payload(budget: TaskLoopBudget) -> dict[str, object]:
    return {
        "max_steps": budget.max_steps,
        "max_tool_calls": budget.max_tool_calls,
        "max_seconds": budget.max_seconds,
    }


def _decision_payload(decision: NextActionDecision) -> dict[str, object]:
    return {
        "action_type": decision.action_type,
        "decision_summary": decision.decision_summary,
        "tool_name": decision.tool_name,
        "arguments": decision.arguments,
        "missing_input": decision.missing_input,
        "risk_level": decision.risk_level,
    }


def _observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "source": observation.source,
        "content": observation.content,
        "created_at": observation.created_at.isoformat(),
    }


def _step_payload(step: ExecutedStep) -> dict[str, object]:
    return {
        "step_number": step.step_number,
        "tool_name": step.tool_name,
        "arguments": step.arguments,
        "status": step.status,
        "decision_summary": step.decision_summary,
        "error": step.error,
        "observation": _observation_payload(step.observation),
    }


def _state_payload(state: TaskState) -> dict[str, object]:
    return {
        "session_id": state.session_id,
        "run_id": state.run_id,
        "user_goal": state.user_goal,
        "status": state.status,
        "observations": [_observation_payload(observation) for observation in state.observations],
        "steps_taken": [_step_payload(step) for step in state.steps_taken],
    }
