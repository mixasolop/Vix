import asyncio

from app.assistant.policy import PolicyEngine
from app.assistant.task_loop import NextActionDecision, TaskLoop, TaskLoopBudget, TaskState
from app.schemas.tools import ConfirmationPolicy, RetryPolicy, RiskLevel, ToolDefinition, ToolResult, ToolStatus
from app.tools.registry import ToolRegistry


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, event_type: str, session_id: str, run_id: str, data: dict[str, object]) -> None:
        self.events.append(
            {
                "type": event_type,
                "session_id": session_id,
                "run_id": run_id,
                "data": data,
            }
        )


def test_task_loop_succeeds_after_two_tool_calls() -> None:
    registry = ToolRegistry()
    calls: list[str] = []

    async def first_tool(arguments: dict[str, object]) -> ToolResult:
        calls.append("first_tool")
        return ToolResult(tool="first_tool", status="success", output={"value": "first observation"})

    async def second_tool(arguments: dict[str, object]) -> ToolResult:
        calls.append("second_tool")
        return ToolResult(tool="second_tool", status="success", output={"value": arguments["input"]})

    registry.register(_tool_definition("first_tool"), first_tool)
    registry.register(_tool_definition("second_tool"), second_tool)

    def decide_next_action(state: TaskState) -> NextActionDecision:
        if len(state.steps_taken) == 0:
            return NextActionDecision(
                action_type="CALL_TOOL",
                decision_summary="Collect the first public observation.",
                tool_name="first_tool",
                arguments={},
                risk_level="READ",
            )
        if len(state.steps_taken) == 1:
            first_observation = state.observations[-1].content
            assert isinstance(first_observation, dict)
            output = first_observation["output"]
            assert isinstance(output, dict)
            return NextActionDecision(
                action_type="CALL_TOOL",
                decision_summary="Use the first observation to call the second tool.",
                tool_name="second_tool",
                arguments={"input": output["value"]},
                risk_level="READ",
            )
        return NextActionDecision(
            action_type="ANSWER",
            decision_summary="The two required observations are available.",
            tool_name=None,
            arguments={"answer": "Task complete."},
        )

    recorder = EventRecorder()
    loop = TaskLoop(
        registry=registry,
        policy_engine=PolicyEngine(),
        decide_next_action=decide_next_action,
        emit_event=recorder.emit,
    )

    state = asyncio.run(loop.run(session_id="sess_1", run_id="run_1", user_goal="run two fake tools"))

    assert state.status == "completed"
    assert calls == ["first_tool", "second_tool"]
    assert [step.tool_name for step in state.steps_taken] == ["first_tool", "second_tool"]
    assert [event["type"] for event in recorder.events] == [
        "task_loop_started",
        "task_step_decided",
        "task_observation_added",
        "task_step_decided",
        "task_observation_added",
        "task_step_decided",
        "task_loop_completed",
    ]
    assert recorder.events[1]["data"]["decision"]["decision_summary"] == "Collect the first public observation."
    assert "reasoning" not in recorder.events[1]["data"]["decision"]
    assert recorder.events[-1]["data"]["stop_reason"] == "answered"


def test_task_loop_asks_clarification_when_input_is_missing() -> None:
    registry = ToolRegistry()

    def decide_next_action(state: TaskState) -> NextActionDecision:
        return NextActionDecision(
            action_type="ASK_USER",
            decision_summary="Need the folder before searching files.",
            tool_name=None,
            missing_input="folder",
        )

    recorder = EventRecorder()
    loop = TaskLoop(
        registry=registry,
        policy_engine=PolicyEngine(),
        decide_next_action=decide_next_action,
        emit_event=recorder.emit,
    )

    state = asyncio.run(loop.run(session_id="sess_1", run_id="run_2", user_goal="find my report"))

    assert state.status == "waiting_for_user"
    assert state.steps_taken == []
    assert [event["type"] for event in recorder.events] == [
        "task_loop_started",
        "task_step_decided",
        "task_loop_completed",
    ]
    assert recorder.events[-1]["data"]["status"] == "waiting_for_user"
    assert recorder.events[-1]["data"]["missing_input"] == "folder"


def test_task_loop_stops_after_max_steps() -> None:
    registry = ToolRegistry()

    async def ping_tool(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(tool="ping_tool", status="success", output={"ok": True})

    registry.register(_tool_definition("ping_tool"), ping_tool)

    def decide_next_action(state: TaskState) -> NextActionDecision:
        return NextActionDecision(
            action_type="CALL_TOOL",
            decision_summary="Try another observation.",
            tool_name="ping_tool",
            arguments={},
            risk_level="READ",
        )

    recorder = EventRecorder()
    loop = TaskLoop(
        registry=registry,
        policy_engine=PolicyEngine(),
        decide_next_action=decide_next_action,
        emit_event=recorder.emit,
        budget=TaskLoopBudget(max_steps=2, max_tool_calls=10, max_seconds=30),
    )

    state = asyncio.run(loop.run(session_id="sess_1", run_id="run_3", user_goal="never stop"))

    assert state.status == "failed"
    assert len(state.steps_taken) == 2
    assert [event["type"] for event in recorder.events].count("task_step_decided") == 2
    assert [event["type"] for event in recorder.events].count("task_observation_added") == 2
    assert recorder.events[-1]["type"] == "task_loop_failed"
    assert recorder.events[-1]["data"]["stop_reason"] == "max_steps"


def test_task_loop_blocks_high_risk_tool_without_permission() -> None:
    registry = ToolRegistry()

    async def high_risk_tool(arguments: dict[str, object]) -> ToolResult:
        raise AssertionError("High-risk tool should not execute without permission.")

    registry.register(
        _tool_definition(
            "send_message",
            risk_level=RiskLevel.high_risk,
            confirmation_policy=ConfirmationPolicy.before_execute,
        ),
        high_risk_tool,
    )

    def decide_next_action(state: TaskState) -> NextActionDecision:
        return NextActionDecision(
            action_type="CALL_TOOL",
            decision_summary="Sending a message would affect an external recipient.",
            tool_name="send_message",
            arguments={"recipient": "Anna", "message": "Hi"},
            risk_level="HIGH_RISK",
        )

    recorder = EventRecorder()
    loop = TaskLoop(
        registry=registry,
        policy_engine=PolicyEngine(),
        decide_next_action=decide_next_action,
        emit_event=recorder.emit,
    )

    state = asyncio.run(loop.run(session_id="sess_1", run_id="run_4", user_goal="message Anna"))

    assert state.status == "blocked"
    assert state.steps_taken == []
    assert [event["type"] for event in recorder.events] == [
        "task_loop_started",
        "task_step_decided",
        "task_loop_failed",
    ]
    failed_event = recorder.events[-1]
    assert failed_event["data"]["stop_reason"] == "blocked_by_policy"
    assert failed_event["data"]["policy_decision"]["requires_permission"] is True
    assert "requires confirmation" in failed_event["data"]["message"]


def _tool_definition(
    name: str,
    risk_level: RiskLevel = RiskLevel.read,
    confirmation_policy: ConfirmationPolicy = ConfirmationPolicy.none,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Fake test tool: {name}",
        status=ToolStatus.implemented,
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        output_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        risk_level=risk_level,
        confirmation_policy=confirmation_policy,
        timeout_seconds=5,
        retry_policy=RetryPolicy(max_attempts=1),
    )
