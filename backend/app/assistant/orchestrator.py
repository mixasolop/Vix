import asyncio
import json
import logging
from uuid import uuid4

from app.assistant.llm_client import DeterministicLLMClient, LLMClient
from app.assistant.planner import FALLBACK_MESSAGE, Planner
from app.assistant.policy import PendingAction, PermissionManager, PolicyEngine
from app.db.database import Database
from app.events.event_bus import EventBus
from app.schemas.chat import ChatAcceptedResponse, ChatRequest
from app.schemas.events import AssistantEvent
from app.schemas.plans import Plan, PlanStep, PlanStepStatus
from app.schemas.tools import PermissionDecisionResponse, PermissionPreview, PermissionRequest, ToolCall, ToolDefinition
from app.tools.registry import ToolRegistry

LOGGER = logging.getLogger("app.orchestrator")


class Orchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        database: Database,
        permission_manager: PermissionManager,
        event_bus: EventBus,
        planner: Planner | None = None,
        policy_engine: PolicyEngine | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self._permission_manager = permission_manager
        self._event_bus = event_bus
        self._planner = planner or Planner()
        self._policy_engine = policy_engine or PolicyEngine()
        self._llm_client = llm_client or DeterministicLLMClient()
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def start_chat(self, request: ChatRequest) -> ChatAcceptedResponse:
        session_id = request.conversation_id or f"sess_{uuid4()}"
        run_id = f"run_{uuid4()}"
        self._database.ensure_session(session_id)
        LOGGER.info(
            "chat accepted | session=%s | run=%s | message=%s",
            session_id,
            run_id,
            _short_text(request.message),
        )

        task = asyncio.create_task(self._run_chat(session_id, run_id, request.message))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

        return ChatAcceptedResponse(accepted=True, conversation_id=session_id, run_id=run_id)

    async def approve_permission(self, permission_id: str) -> PermissionDecisionResponse:
        response, pending_action = self._permission_manager.approve(permission_id)
        LOGGER.info(
            "permission approved | permission=%s | resumes_action=%s",
            permission_id,
            pending_action is not None,
        )
        await self._emit(
            "permission_approved",
            pending_action.session_id if pending_action else None,
            pending_action.run_id if pending_action else None,
            {"permission_id": permission_id},
        )
        if pending_action is not None:
            task = asyncio.create_task(self._execute_pending_action(pending_action))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
        return response

    async def reject_permission(self, permission_id: str) -> PermissionDecisionResponse:
        response, pending_action = self._permission_manager.reject(permission_id)
        LOGGER.info(
            "permission rejected | permission=%s | cancels_action=%s",
            permission_id,
            pending_action is not None,
        )
        await self._emit(
            "permission_rejected",
            pending_action.session_id if pending_action else None,
            pending_action.run_id if pending_action else None,
            {"permission_id": permission_id},
        )
        if pending_action is not None:
            assistant_message = "Action canceled."
            self._database.log_message(pending_action.session_id, "assistant", assistant_message)
            await self._emit(
                "assistant_message_created",
                pending_action.session_id,
                pending_action.run_id,
                {"role": "assistant", "message": assistant_message},
            )
        return response

    async def _run_chat(self, session_id: str, run_id: str, message: str) -> None:
        try:
            normalized_message = message.strip()
            LOGGER.info("run started | session=%s | run=%s", session_id, run_id)
            self._database.log_message(session_id, "user", normalized_message)
            await self._emit(
                "user_message_received",
                session_id,
                run_id,
                {"role": "user", "message": normalized_message},
            )
            await self._emit("assistant_thinking_started", session_id, run_id, {})

            plan = self._planner.create_plan(normalized_message)
            await self._emit_plan("plan_created", session_id, run_id, plan)

            proposal = self._planner.propose_tool_call(normalized_message)
            if proposal is None:
                LOGGER.info("planner produced no tool call | session=%s | run=%s", session_id, run_id)
                assistant_message = await self._llm_client.complete([{"role": "user", "content": normalized_message}])
                await self._finish_without_tool(session_id, run_id, assistant_message or FALLBACK_MESSAGE)
                return

            metadata = self._registry.get(proposal.name)
            if metadata is None:
                LOGGER.error("planner selected unregistered tool | tool=%s | session=%s | run=%s", proposal.name, session_id, run_id)
                await self._fail_run(session_id, run_id, f"Tool is not registered: {proposal.name}.")
                return

            policy_decision = self._policy_engine.evaluate_tool_call(metadata, proposal.arguments)
            if policy_decision.blocked:
                LOGGER.warning(
                    "policy blocked tool call | tool=%s | session=%s | run=%s | reason=%s",
                    proposal.name,
                    session_id,
                    run_id,
                    policy_decision.reason,
                )
                await self._fail_run(session_id, run_id, policy_decision.reason)
                return

            selected_plan = self._plan_with_statuses(plan, select_status=PlanStepStatus.completed, execute_status=PlanStepStatus.pending)
            await self._emit(
                "tool_selected",
                session_id,
                run_id,
                {
                    "tool_name": proposal.name,
                    "arguments": proposal.arguments,
                    "risk_level": metadata.risk_level.value,
                    "confirmation_policy": metadata.confirmation_policy.value,
                    "policy_decision": {
                        "allowed": policy_decision.allowed,
                        "requires_permission": policy_decision.requires_permission,
                        "blocked": policy_decision.blocked,
                        "reason": policy_decision.reason,
                    },
                    "plan": selected_plan.model_dump(mode="json"),
                },
            )

            if policy_decision.requires_permission:
                permission_id = str(uuid4())
                LOGGER.info(
                    "permission required | permission=%s | tool=%s | session=%s | run=%s",
                    permission_id,
                    proposal.name,
                    session_id,
                    run_id,
                )
                permission = PermissionRequest(
                    permission_id=permission_id,
                    tool=proposal.name,
                    reason=policy_decision.reason,
                    arguments=proposal.arguments,
                )
                preview = self._build_permission_preview(
                    permission_id=permission_id,
                    tool=metadata,
                    arguments=proposal.arguments,
                    reason=permission.reason,
                ).model_dump(mode="json")
                self._database.create_permission(
                    permission_id=permission_id,
                    session_id=session_id,
                    action_type=proposal.name,
                    preview=preview,
                )
                self._permission_manager.add_pending_action(
                    PendingAction(
                        permission_id=permission_id,
                        session_id=session_id,
                        run_id=run_id,
                        tool_name=proposal.name,
                        arguments=proposal.arguments,
                    )
                )
                await self._emit(
                    "permission_required",
                    session_id,
                    run_id,
                    {
                        "permission_id": permission_id,
                        "action_type": proposal.name,
                        "preview": preview,
                        "permission": permission.model_dump(mode="json"),
                    },
                )
                return

            if not policy_decision.allowed:
                LOGGER.error(
                    "policy returned non-executable decision | tool=%s | session=%s | run=%s | reason=%s",
                    proposal.name,
                    session_id,
                    run_id,
                    policy_decision.reason,
                )
                await self._fail_run(session_id, run_id, policy_decision.reason)
                return

            await self._execute_tool(session_id, run_id, proposal.name, proposal.arguments, selected_plan)
        except Exception as exc:
            LOGGER.exception("run failed with unhandled exception | session=%s | run=%s", session_id, run_id)
            await self._fail_run(session_id, run_id, str(exc))

    async def _execute_pending_action(self, action: PendingAction) -> None:
        plan = self._plan_with_statuses(
            Plan(
                goal=f"Run approved action: {action.tool_name}",
                steps=[
                    PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Select tool", status=PlanStepStatus.completed),
                    PlanStep(number=3, title="Execute", status=PlanStepStatus.pending),
                ],
            ),
            select_status=PlanStepStatus.completed,
            execute_status=PlanStepStatus.pending,
        )
        await self._execute_tool(action.session_id, action.run_id, action.tool_name, action.arguments, plan)

    async def _execute_tool(
        self,
        session_id: str,
        run_id: str,
        tool_name: str,
        arguments: dict[str, object],
        selected_plan: Plan,
    ) -> None:
        running_plan = self._plan_with_statuses(selected_plan, select_status=PlanStepStatus.completed, execute_status=PlanStepStatus.running)
        await self._emit(
            "tool_started",
            session_id,
            run_id,
            {"tool_name": tool_name, "arguments": arguments, "plan": running_plan.model_dump(mode="json")},
        )

        result = await self._registry.execute(tool_name, arguments)
        LOGGER.info(
            "tool execution finished | tool=%s | status=%s | session=%s | run=%s",
            tool_name,
            result.status,
            session_id,
            run_id,
        )
        self._database.log_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            status=result.status,
            result=result.output,
            error=result.error,
        )

        final_status = PlanStepStatus.completed if result.status == "success" else PlanStepStatus.failed
        final_plan = self._plan_with_statuses(selected_plan, select_status=PlanStepStatus.completed, execute_status=final_status)
        tool_call = ToolCall(tool=tool_name, arguments=arguments, status=result.status, result=result.output, error=result.error)
        await self._emit(
            "tool_result",
            session_id,
            run_id,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "status": result.status,
                "result": result.output,
                "error": result.error,
                "tool_call": tool_call.model_dump(mode="json"),
                "plan": final_plan.model_dump(mode="json"),
            },
        )

        assistant_message = self._assistant_message_for_tool_result(tool_call)
        self._database.log_message(session_id, "assistant", assistant_message)
        await self._emit(
            "assistant_message_created",
            session_id,
            run_id,
            {"role": "assistant", "message": assistant_message},
        )

    async def _finish_without_tool(self, session_id: str, run_id: str, assistant_message: str) -> None:
        self._database.log_message(session_id, "assistant", assistant_message)
        await self._emit("assistant_message_created", session_id, run_id, {"role": "assistant", "message": assistant_message})

    async def _fail_run(self, session_id: str, run_id: str, message: str) -> None:
        await self._emit("error_occurred", session_id, run_id, {"message": message})
        self._database.log_message(session_id, "assistant", message)
        await self._emit("assistant_message_created", session_id, run_id, {"role": "assistant", "message": message})

    async def _emit_plan(self, event_type: str, session_id: str, run_id: str, plan: Plan) -> None:
        await self._emit(event_type, session_id, run_id, {"plan": plan.model_dump(mode="json")})

    async def _emit(self, event_type: str, session_id: str | None, run_id: str | None, data: dict[str, object]) -> None:
        event = AssistantEvent(type=event_type, session_id=session_id, run_id=run_id, data=data)
        self._database.log_event(event)
        LOGGER.info(
            "event emitted | type=%s | session=%s | run=%s | data=%s",
            event_type,
            session_id or "-",
            run_id or "-",
            _json_for_log(data),
        )
        await self._event_bus.publish(event)

    @staticmethod
    def _plan_with_statuses(plan: Plan, select_status: PlanStepStatus, execute_status: PlanStepStatus) -> Plan:
        updated_steps: list[PlanStep] = []
        for step in plan.steps:
            if step.number == 2:
                updated_steps.append(PlanStep(number=step.number, title=step.title, status=select_status))
            elif step.number == 3:
                updated_steps.append(PlanStep(number=step.number, title=step.title, status=execute_status))
            else:
                updated_steps.append(step)
        return Plan(goal=plan.goal, steps=updated_steps)

    @staticmethod
    def _assistant_message_for_tool_result(tool_call: ToolCall) -> str:
        if tool_call.status == "success":
            message = tool_call.result.get("message")
            return str(message) if message else f"{tool_call.tool} completed."
        return tool_call.error or f"{tool_call.tool} failed."

    @staticmethod
    def _build_permission_preview(
        permission_id: str,
        tool: ToolDefinition,
        arguments: dict[str, object],
        reason: str,
    ) -> PermissionPreview:
        target = _preview_target(tool.name, arguments)
        return PermissionPreview(
            permission_id=permission_id,
            tool_name=tool.name,
            action=f"Run {tool.name}",
            target=target,
            content=dict(arguments),
            risk_level=tool.risk_level,
            what_will_happen=_preview_what_will_happen(tool.name, target),
            reason=reason,
            editable=False,
            edit_schema=tool.input_schema,
            arguments=dict(arguments),
        )


def _json_for_log(data: dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _short_text(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _preview_target(tool_name: str, arguments: dict[str, object]) -> str:
    if tool_name == "launch_app":
        return str(arguments.get("app_name", "")).strip() or "Unknown application"
    if tool_name == "send_message":
        return str(arguments.get("recipient", "")).strip() or "Unknown recipient"
    if tool_name == "pay_for_order":
        return str(arguments.get("order_id", "")).strip() or "Unknown order"
    if "path" in arguments:
        return str(arguments["path"])
    if "url" in arguments:
        return str(arguments["url"])
    return tool_name


def _preview_what_will_happen(tool_name: str, target: str) -> str:
    if tool_name == "launch_app":
        return f"The assistant will launch the whitelisted Windows application '{target}'."
    if tool_name == "send_message":
        return f"The assistant will send the shown message to '{target}' after approval."
    if tool_name == "pay_for_order":
        return f"The assistant will submit payment for order '{target}' after approval."
    return f"The assistant will run '{tool_name}' with the shown content."
