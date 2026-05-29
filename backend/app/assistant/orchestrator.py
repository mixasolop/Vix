import asyncio
import json
import logging
from uuid import uuid4

from app.assistant.llm_client import DeterministicLLMClient, LLMClient
from app.assistant.planner import FALLBACK_MESSAGE, Planner
from app.assistant.policy import PendingAction, PermissionManager, PolicyEngine
from app.assistant.request_classifier import (
    RequestCategory,
    RequestClassification,
    classify_request,
    should_propose_missing_tool_after_answer,
)
from app.context.window_tracker import WindowTracker
from app.db.database import Database
from app.events.event_bus import EventBus
from app.schemas.chat import ChatAcceptedResponse, ChatRequest
from app.schemas.events import AssistantEvent
from app.schemas.plans import Plan, PlanStep, PlanStepStatus
from app.schemas.proposed_tools import CreateProposedToolRequest, ProposedTool
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
        context_tracker: WindowTracker | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self._permission_manager = permission_manager
        self._event_bus = event_bus
        self._planner = planner or Planner()
        self._policy_engine = policy_engine or PolicyEngine()
        self._llm_client = llm_client or DeterministicLLMClient()
        self._context_tracker = context_tracker
        self._active_tasks: set[asyncio.Task[None]] = set()

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    def set_llm_client(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

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

            classification = classify_request(normalized_message, self._planner)
            LOGGER.info(
                "request classified | session=%s | run=%s | category=%s | reason=%s",
                session_id,
                run_id,
                classification.category.value,
                classification.reason,
            )
            await self._emit(
                "request_classified",
                session_id,
                run_id,
                {
                    "category": classification.category.value,
                    "reason": classification.reason,
                    "tool_name": classification.tool_proposal.name if classification.tool_proposal else None,
                    "arguments": classification.tool_proposal.arguments if classification.tool_proposal else {},
                    "missing_input": classification.missing_input,
                },
            )
            plan = self._create_plan_for_request(normalized_message, classification)
            LOGGER.info(
                "planner plan created | session=%s | run=%s | plan=%s",
                session_id,
                run_id,
                plan.model_dump_json(),
            )
            await self._emit_plan("plan_created", session_id, run_id, plan)

            if classification.category == RequestCategory.general_answer:
                await self._answer_with_llm(session_id, run_id, normalized_message)
                return

            if classification.category == RequestCategory.realtime_info:
                await self._handle_realtime_info(session_id, run_id, classification, plan)
                return

            if classification.category == RequestCategory.local_context:
                await self._handle_local_context(session_id, run_id, normalized_message, classification, plan)
                return

            if classification.category == RequestCategory.missing_tool:
                proposed_tool = await self._propose_missing_tool(session_id, run_id, normalized_message)
                if proposed_tool is not None:
                    assistant_message = f"I cannot do this yet, but I proposed a new tool: {proposed_tool.name}."
                    await self._finish_without_tool(session_id, run_id, assistant_message)
                    return
                await self._finish_without_tool(session_id, run_id, FALLBACK_MESSAGE)
                return

            if classification.category == RequestCategory.unsafe_or_blocked:
                await self._fail_run(session_id, run_id, classification.reason)
                return

            proposal = classification.tool_proposal
            if proposal is None:
                LOGGER.info("planner produced no tool call | session=%s | run=%s", session_id, run_id)
                proposed_tool = await self._propose_missing_tool(session_id, run_id, normalized_message)
                if proposed_tool is not None:
                    assistant_message = f"I cannot do this yet, but I proposed a new tool: {proposed_tool.name}."
                    await self._finish_without_tool(session_id, run_id, assistant_message)
                    return

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

    async def _answer_with_llm(self, session_id: str, run_id: str, user_message: str) -> None:
        messages = self._recent_session_messages(session_id)
        await self._emit(
            "llm_response_started",
            session_id,
            run_id,
            {"message_count": len(messages), "purpose": "general_answer"},
        )
        try:
            assistant_message = await self._llm_client.complete(messages)
        except Exception as exc:
            LOGGER.exception("llm completion failed | session=%s | run=%s", session_id, run_id)
            await self._emit(
                "llm_response_finished",
                session_id,
                run_id,
                {"status": "failed", "error": str(exc), "purpose": "general_answer"},
            )
            await self._finish_without_tool(session_id, run_id, "I could not get an AI answer right now. Please try again.")
            return

        await self._emit(
            "llm_response_finished",
            session_id,
            run_id,
            {"status": "success", "purpose": "general_answer", "message": assistant_message},
        )
        if should_propose_missing_tool_after_answer(user_message, assistant_message):
            proposed_tool = await self._propose_missing_tool(session_id, run_id, user_message)
            if proposed_tool is not None:
                assistant_message = (
                    f"{assistant_message}\n\n"
                    f"I also proposed a new tool for developer review: {proposed_tool.name}."
                )

        await self._finish_without_tool(session_id, run_id, assistant_message or FALLBACK_MESSAGE)

    async def _handle_realtime_info(
        self,
        session_id: str,
        run_id: str,
        classification: RequestClassification,
        plan: Plan,
    ) -> None:
        if classification.missing_input == "location":
            await self._finish_without_tool(session_id, run_id, "Which city should I check?")
            return

        if classification.tool_proposal is None:
            await self._finish_without_tool(session_id, run_id, "I need a real-time information tool for that, but I cannot map this request yet.")
            return

        selected_plan = self._plan_with_statuses(plan, select_status=PlanStepStatus.completed, execute_status=PlanStepStatus.pending)
        await self._execute_classified_tool(session_id, run_id, classification.tool_proposal, selected_plan)

    async def _handle_local_context(
        self,
        session_id: str,
        run_id: str,
        user_message: str,
        classification: RequestClassification,
        plan: Plan,
    ) -> None:
        if classification.tool_proposal is None:
            await self._finish_without_tool(session_id, run_id, "I could not map that context request to a context tool.")
            return

        context_window = None
        if classification.tool_proposal.name in {"get_selected_text", "get_context_window_info"}:
            context_window = self._context_window_data()
            await self._emit(
                "context_window_selected",
                session_id,
                run_id,
                {"window": context_window or {}, "tool_name": classification.tool_proposal.name},
            )

        selected_plan = self._plan_with_statuses(plan, select_status=PlanStepStatus.completed, execute_status=PlanStepStatus.pending)
        tool_call = await self._execute_context_tool(session_id, run_id, classification.tool_proposal, selected_plan)
        if tool_call is None:
            return

        if tool_call.status != "success":
            await self._finish_without_tool(session_id, run_id, tool_call.error or f"{tool_call.tool} failed.")
            return

        await self._emit(
            "context_captured",
            session_id,
            run_id,
            {
                "tool_name": tool_call.tool,
                "artifact_type": _artifact_type_for_tool(tool_call.tool),
                "content_preview": _context_content_preview(tool_call),
                "context_window": tool_call.result.get("context_window") or tool_call.result.get("window") or {},
            },
        )

        artifact = self._create_context_artifact(session_id, run_id, tool_call)
        if artifact is not None:
            await self._emit(
                "artifact_created",
                session_id,
                run_id,
                {
                    "artifact": {
                        "id": artifact.id,
                        "type": artifact.type,
                        "title": artifact.title,
                        "content_text": artifact.content_text,
                        "data": json.loads(artifact.data_json),
                        "created_at": artifact.created_at.isoformat(),
                    }
                },
            )

        await self._answer_with_context(session_id, run_id, user_message, tool_call)

    async def _answer_with_context(self, session_id: str, run_id: str, user_message: str, tool_call: ToolCall) -> None:
        context_prompt = _context_prompt(user_message, tool_call)
        messages = self._recent_session_messages(session_id)
        messages.append({"role": "user", "content": context_prompt})
        await self._emit(
            "llm_response_started",
            session_id,
            run_id,
            {"message_count": len(messages), "purpose": "local_context"},
        )
        try:
            assistant_message = await self._llm_client.complete(messages)
        except Exception as exc:
            LOGGER.exception("context llm completion failed | session=%s | run=%s", session_id, run_id)
            await self._emit(
                "llm_response_finished",
                session_id,
                run_id,
                {"status": "failed", "error": str(exc), "purpose": "local_context"},
            )
            await self._finish_without_tool(session_id, run_id, "I captured the context, but I could not get an AI answer right now.")
            return

        await self._emit(
            "llm_response_finished",
            session_id,
            run_id,
            {"status": "success", "purpose": "local_context", "message": assistant_message},
        )
        final_message = assistant_message
        if not final_message or final_message == FALLBACK_MESSAGE:
            final_message = self._assistant_message_for_tool_result(tool_call)
        await self._finish_without_tool(session_id, run_id, final_message)

    async def _execute_context_tool(
        self,
        session_id: str,
        run_id: str,
        proposal,
        selected_plan: Plan,
    ) -> ToolCall | None:
        metadata = self._registry.get(proposal.name)
        if metadata is None:
            await self._fail_run(session_id, run_id, f"Tool is not registered: {proposal.name}.")
            return None

        policy_decision = self._policy_engine.evaluate_tool_call(metadata, proposal.arguments)
        if policy_decision.blocked:
            await self._fail_run(session_id, run_id, policy_decision.reason)
            return None

        await self._emit(
            "tool_selected",
            session_id,
            run_id,
            {
                "tool_name": proposal.name,
                "arguments": proposal.arguments,
                "risk_level": metadata.risk_level.value,
                "confirmation_policy": metadata.confirmation_policy.value,
                "privacy_sensitive": proposal.name in {"get_clipboard_text", "get_selected_text"},
                "policy_decision": {
                    "allowed": policy_decision.allowed,
                    "requires_permission": policy_decision.requires_permission,
                    "blocked": policy_decision.blocked,
                    "reason": policy_decision.reason,
                },
                "plan": selected_plan.model_dump(mode="json"),
            },
        )

        if not policy_decision.allowed:
            await self._fail_run(session_id, run_id, policy_decision.reason)
            return None

        running_plan = self._plan_with_statuses(selected_plan, select_status=PlanStepStatus.completed, execute_status=PlanStepStatus.running)
        await self._emit(
            "tool_started",
            session_id,
            run_id,
            {"tool_name": proposal.name, "arguments": proposal.arguments, "plan": running_plan.model_dump(mode="json")},
        )

        result = await self._registry.execute(proposal.name, proposal.arguments)
        self._database.log_tool_call(
            session_id=session_id,
            tool_name=proposal.name,
            arguments=proposal.arguments,
            status=result.status,
            result=result.output,
            error=result.error,
        )
        final_status = PlanStepStatus.completed if result.status == "success" else PlanStepStatus.failed
        final_plan = self._plan_with_statuses(selected_plan, select_status=PlanStepStatus.completed, execute_status=final_status)
        tool_call = ToolCall(tool=proposal.name, arguments=proposal.arguments, status=result.status, result=result.output, error=result.error)
        await self._emit(
            "tool_result",
            session_id,
            run_id,
            {
                "tool_name": proposal.name,
                "arguments": proposal.arguments,
                "status": result.status,
                "result": result.output,
                "error": result.error,
                "tool_call": tool_call.model_dump(mode="json"),
                "plan": final_plan.model_dump(mode="json"),
            },
        )
        return tool_call

    async def _execute_classified_tool(
        self,
        session_id: str,
        run_id: str,
        proposal,
        selected_plan: Plan,
    ) -> None:
        metadata = self._registry.get(proposal.name)
        if metadata is None:
            LOGGER.error("classifier selected unregistered tool | tool=%s | session=%s | run=%s", proposal.name, session_id, run_id)
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

    def _create_plan_for_request(self, user_message: str, classification: RequestClassification) -> Plan:
        if classification.category == RequestCategory.general_answer:
            return Plan(
                goal=f"Answer: {user_message}",
                steps=[
                    PlanStep(number=1, title="Understand question", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Generate LLM answer", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.realtime_info:
            return Plan(
                goal=f"Fetch real-time information: {user_message}",
                steps=[
                    PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Select real-time tool", status=PlanStepStatus.pending),
                    PlanStep(number=3, title="Execute", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.local_context:
            return Plan(
                goal=f"Capture local context for: {user_message}",
                steps=[
                    PlanStep(number=1, title="Understand context request", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Select context tool", status=PlanStepStatus.pending),
                    PlanStep(number=3, title="Capture context", status=PlanStepStatus.pending),
                    PlanStep(number=4, title="Answer using captured context", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.missing_tool:
            return Plan(
                goal=f"Handle missing capability: {user_message}",
                steps=[
                    PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Propose missing tool", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.unsafe_or_blocked:
            return Plan(
                goal=f"Block unsafe request: {user_message}",
                steps=[
                    PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, title="Apply safety policy", status=PlanStepStatus.pending),
                ],
            )

        return self._planner.create_plan(user_message)

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
        if result.status != "success":
            self._database.create_reflection(
                session_id=session_id,
                run_id=run_id,
                source_type="failed_tool",
                note=f"Tool {tool_name} failed with error: {result.error or 'unknown error'}.",
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

    def _create_context_artifact(self, session_id: str, run_id: str, tool_call: ToolCall):
        artifact_type = _artifact_type_for_tool(tool_call.tool)
        if artifact_type is None:
            return None

        content_text = tool_call.result.get("text")
        content_text = str(content_text) if content_text is not None else None
        title = _artifact_title(tool_call)
        data = {
            "tool": tool_call.tool,
            "arguments": tool_call.arguments,
            "result": tool_call.result,
            "dev_mode_full_text_stored": True,
        }
        return self._database.create_artifact(
            session_id=session_id,
            run_id=run_id,
            artifact_type=artifact_type,
            title=title,
            content_text=content_text,
            data=data,
        )

    def _context_window_data(self) -> dict[str, object] | None:
        if self._context_tracker is None:
            return None
        window = self._context_tracker.get_context_window(validate_exists=False)
        return window.model_dump(mode="json") if window is not None else None

    def _recent_session_messages(self, session_id: str, limit: int = 20) -> list[dict[str, object]]:
        records = self._database.list_messages(session_id)[-limit:]
        messages: list[dict[str, object]] = []
        for record in records:
            if record.role not in {"user", "assistant", "system"}:
                continue
            messages.append({"role": record.role, "content": record.content})
        return messages

    async def _propose_missing_tool(self, session_id: str, run_id: str, user_message: str) -> ProposedTool | None:
        draft = await self._llm_client.propose_tool_spec(user_message, self._registry.list_tools())
        if draft is None:
            return None

        LOGGER.info(
            "missing capability proposed tool draft | session=%s | run=%s | draft=%s",
            session_id,
            run_id,
            draft.model_dump_json(),
        )

        proposed_tool = self._database.create_proposed_tool(
            CreateProposedToolRequest(
                name=draft.name,
                description=draft.description,
                reason=draft.reason,
                risk_level=draft.risk_level,
                input_schema=draft.input_schema,
                output_schema=draft.output_schema,
                created_from_message=user_message,
            )
        )
        self._database.create_reflection(
            session_id=session_id,
            run_id=run_id,
            source_type="missing_tool",
            note=f"User asked for unsupported capability. Proposed {proposed_tool.name}: {proposed_tool.reason}",
        )
        await self._emit_proposed_tool("proposed_tool_created", session_id, run_id, proposed_tool)
        return proposed_tool

    async def _fail_run(self, session_id: str, run_id: str, message: str) -> None:
        await self._emit("error_occurred", session_id, run_id, {"message": message})
        self._database.log_message(session_id, "assistant", message)
        await self._emit("assistant_message_created", session_id, run_id, {"role": "assistant", "message": message})

    async def _emit_proposed_tool(self, event_type: str, session_id: str, run_id: str, proposed_tool: ProposedTool) -> None:
        await self._emit(
            event_type,
            session_id,
            run_id,
            {
                "tool_id": proposed_tool.id,
                "name": proposed_tool.name,
                "reason": proposed_tool.reason,
                "risk_level": proposed_tool.risk_level,
                "status": proposed_tool.status.value,
                "tool": proposed_tool.model_dump(mode="json"),
            },
        )

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


def _artifact_type_for_tool(tool_name: str) -> str | None:
    return {
        "get_clipboard_text": "clipboard_text",
        "get_selected_text": "selected_text",
        "get_context_window_info": "context_window_info",
        "get_foreground_window_info": "foreground_window_info",
    }.get(tool_name)


def _artifact_title(tool_call: ToolCall) -> str:
    if tool_call.tool == "get_selected_text":
        window = tool_call.result.get("context_window", {})
        if isinstance(window, dict):
            title = window.get("title") or window.get("process_name") or "context window"
            return f"Selected text from {title}"
        return "Selected text"
    if tool_call.tool == "get_clipboard_text":
        return "Clipboard text"
    if tool_call.tool == "get_context_window_info":
        return "Context window info"
    if tool_call.tool == "get_foreground_window_info":
        return "Foreground window info"
    return tool_call.tool


def _context_content_preview(tool_call: ToolCall) -> str:
    text = tool_call.result.get("text")
    if text is not None:
        return _short_text(str(text), limit=160)
    message = tool_call.result.get("message")
    if message is not None:
        return _short_text(str(message), limit=160)
    return ""


def _context_prompt(user_message: str, tool_call: ToolCall) -> str:
    if tool_call.tool in {"get_selected_text", "get_clipboard_text"}:
        text = str(tool_call.result.get("text", ""))
        source = "selected text" if tool_call.tool == "get_selected_text" else "clipboard"
        window = tool_call.result.get("context_window")
        window_line = ""
        if isinstance(window, dict):
            window_line = f"\nSource window: {window.get('title') or ''} ({window.get('process_name') or 'unknown process'})."
        return (
            f'User asked: "{user_message}"\n'
            f"Captured {source}:{window_line}\n"
            f'"""\n{text}\n"""\n'
            "Answer using only this captured local context plus general knowledge. "
            "Do not claim you captured anything else."
        )

    if tool_call.tool in {"get_context_window_info", "get_foreground_window_info"}:
        return (
            f'User asked: "{user_message}"\n'
            f"Captured window information:\n{json.dumps(tool_call.result, ensure_ascii=False, indent=2, default=str)}\n"
            "Answer directly from this window information. Distinguish context window from technical foreground if relevant."
        )

    return (
        f'User asked: "{user_message}"\n'
        f"Tool result:\n{json.dumps(tool_call.result, ensure_ascii=False, indent=2, default=str)}"
    )
