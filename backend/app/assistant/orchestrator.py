import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
from uuid import uuid4

from app.assistant.context_intent import ContextIntent, ContextOperation, ContextSource, resolve_context_intent
from app.assistant.llm_client import DeterministicLLMClient, LLMClient
from app.assistant.planner import FALLBACK_MESSAGE, Planner, ToolProposal
from app.assistant.policy import PendingAction, PermissionManager, PolicyEngine
from app.assistant.request_classifier import (
    RequestCategory,
    RequestClassification,
    classify_request,
    should_propose_missing_tool_after_answer,
)
from app.assistant.semantic_router import SemanticRouter, SemanticRouteValidationError, ValidatedSemanticRoute
from app.assistant.task_loop import NextActionDecision, TaskLoop, TaskState
from app.assistant.text_canonicalization import canonicalize_user_text
from app.context.window_tracker import WindowTracker
from app.db.models import ClarificationRecord
from app.db.database import Database
from app.events.event_bus import EventBus
from app.schemas.chat import ChatAcceptedResponse, ChatRequest
from app.schemas.events import AssistantEvent
from app.schemas.plans import Plan, PlanStep, PlanStepStatus, update_plan_step, update_plan_steps
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
        semantic_router: SemanticRouter | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self._permission_manager = permission_manager
        self._event_bus = event_bus
        self._planner = planner or Planner()
        self._policy_engine = policy_engine or PolicyEngine()
        self._llm_client = llm_client or DeterministicLLMClient()
        self._context_tracker = context_tracker
        self._semantic_router = semantic_router
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

    async def _try_resolve_pending_clarification(self, session_id: str, run_id: str, message: str) -> bool:
        expired_records = self._database.expire_due_clarifications(session_id)
        for record in expired_records:
            await self._emit("clarification_expired", session_id, run_id, _clarification_payload(record))

        reply_kind = _clarification_reply_kind(message)
        if expired_records and reply_kind is not None:
            await self._finish_without_tool(session_id, run_id, "That clarification expired. Please repeat the request.")
            return True

        pending = self._database.get_pending_clarification(session_id)
        if pending is None or reply_kind is None:
            return False

        if reply_kind == "reject":
            updated = self._database.update_clarification_status(pending.id, "rejected") or pending
            await self._emit("clarification_rejected", session_id, run_id, _clarification_payload(updated))
            await self._finish_without_tool(session_id, run_id, "Okay, no action taken.")
            return True

        updated = self._database.update_clarification_status(pending.id, "accepted") or pending
        await self._emit("clarification_accepted", session_id, run_id, _clarification_payload(updated))
        if pending.proposed_tool_name is None:
            await self._finish_without_tool(session_id, run_id, "Okay, no action taken.")
            return True

        proposal = ToolProposal(name=pending.proposed_tool_name, arguments=_clarification_arguments(pending))
        metadata = self._registry.get(proposal.name)
        if metadata is None:
            await self._fail_run(session_id, run_id, f"Tool is not registered: {proposal.name}.")
            return True

        policy_decision = self._policy_engine.evaluate_tool_call(metadata, proposal.arguments)
        plan = Plan(
            goal=f"Run clarified action: {pending.question}",
            steps=[
                PlanStep(number=1, key="resolve_clarification", title="Resolve clarification", status=PlanStepStatus.completed),
                PlanStep(number=2, key="select_tool", title="Select tool", status=PlanStepStatus.pending),
                PlanStep(number=3, key="execute_tool", title="Execute tool", status=PlanStepStatus.pending),
            ],
        )
        await self._emit_plan("plan_created", session_id, run_id, plan)
        selected_plan = self._tool_selected_plan(
            plan=plan,
            tool_name=proposal.name,
            risk_level=metadata.risk_level.value,
            requires_permission=policy_decision.requires_permission,
        )
        await self._execute_classified_tool(session_id, run_id, proposal, selected_plan)
        return True

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

            if await self._try_resolve_pending_clarification(session_id, run_id, normalized_message):
                return

            classification = classify_request(normalized_message, self._planner)
            semantic_route = await self._maybe_route_semantically(normalized_message, classification)
            if semantic_route is not None:
                classification = semantic_route.classification
                await self._emit(
                    "semantic_route_decided",
                    session_id,
                    run_id,
                    {
                        "original_message": normalized_message,
                        "canonical_message": canonicalize_user_text(normalized_message),
                        "selected_category": semantic_route.route.category,
                        "selected_tool": semantic_route.route.tool_name,
                        "confidence": semantic_route.route.confidence,
                        "reason_summary": semantic_route.route.reason_summary,
                        "router_source": "semantic",
                        "policy_decision": (
                            {
                                "allowed": semantic_route.policy_decision.allowed,
                                "requires_permission": semantic_route.policy_decision.requires_permission,
                                "blocked": semantic_route.policy_decision.blocked,
                                "reason": semantic_route.policy_decision.reason,
                            }
                            if semantic_route.policy_decision is not None
                            else None
                        ),
                    },
                )
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
                    "intent_type": classification.action_decision.intent_type if classification.action_decision else None,
                    "intent_confidence": classification.action_decision.confidence if classification.action_decision else None,
                    "context_intent": _context_intent_payload(classification.context_intent)
                    if classification.context_intent is not None
                    else None,
                    "canonical_message": classification.canonical_message,
                    "router_source": classification.router_source,
                    "tool_sequence": [
                        {"tool_name": proposal.name, "arguments": proposal.arguments}
                        for proposal in classification.tool_sequence
                    ],
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
                await self._answer_with_llm(session_id, run_id, normalized_message, plan)
                return

            if classification.category == RequestCategory.no_action:
                await self._finish_without_tool(session_id, run_id, classification.reason)
                return

            if classification.category == RequestCategory.ask_clarification:
                await self._handle_clarification_required(session_id, run_id, classification)
                return

            if classification.category == RequestCategory.realtime_info:
                await self._handle_realtime_info(session_id, run_id, classification, plan)
                return

            if classification.category == RequestCategory.local_context:
                await self._handle_local_context(session_id, run_id, normalized_message, classification, plan)
                return

            if classification.category == RequestCategory.multi_step:
                await self._handle_multi_step_task(session_id, run_id, normalized_message, classification)
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

            selected_plan = self._tool_selected_plan(
                plan=plan,
                tool_name=proposal.name,
                risk_level=metadata.risk_level.value,
                requires_permission=policy_decision.requires_permission,
            )
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

    async def _answer_with_llm(self, session_id: str, run_id: str, user_message: str, plan: Plan) -> None:
        running_plan = update_plan_step(plan, 2, PlanStepStatus.running)
        messages = self._recent_session_messages(session_id)
        await self._emit(
            "llm_response_started",
            session_id,
            run_id,
            {"message_count": len(messages), "purpose": "general_answer", "plan": running_plan.model_dump(mode="json")},
        )
        try:
            assistant_message = await self._llm_client.complete(messages)
        except Exception as exc:
            LOGGER.exception("llm completion failed | session=%s | run=%s", session_id, run_id)
            failed_plan = update_plan_step(running_plan, 2, PlanStepStatus.failed)
            await self._emit(
                "llm_response_finished",
                session_id,
                run_id,
                {"status": "failed", "error": str(exc), "purpose": "general_answer", "plan": failed_plan.model_dump(mode="json")},
            )
            await self._finish_without_tool(session_id, run_id, "I could not get an AI answer right now. Please try again.")
            return

        completed_plan = update_plan_step(running_plan, 2, PlanStepStatus.completed)
        await self._emit(
            "llm_response_finished",
            session_id,
            run_id,
            {"status": "success", "purpose": "general_answer", "message": assistant_message, "plan": completed_plan.model_dump(mode="json")},
        )
        if should_propose_missing_tool_after_answer(user_message, assistant_message):
            proposed_tool = await self._propose_missing_tool(session_id, run_id, user_message)
            if proposed_tool is not None:
                assistant_message = (
                    f"{assistant_message}\n\n"
                    f"I also proposed a new tool for developer review: {proposed_tool.name}."
                )

        await self._finish_without_tool(session_id, run_id, assistant_message or FALLBACK_MESSAGE)

    async def _maybe_route_semantically(
        self,
        user_message: str,
        deterministic_classification: RequestClassification,
    ) -> ValidatedSemanticRoute | None:
        if self._semantic_router is None:
            return None
        if deterministic_classification.category != RequestCategory.general_answer:
            return None
        try:
            return await self._semantic_router.route(
                original_message=user_message,
                canonical_message=canonicalize_user_text(user_message),
            )
        except SemanticRouteValidationError as exc:
            LOGGER.warning("semantic router rejected route | message=%s | error=%s", _short_text(user_message), exc)
            return None

    async def _handle_clarification_required(
        self,
        session_id: str,
        run_id: str,
        classification: RequestClassification,
    ) -> None:
        question = classification.reason or "Can you clarify what you want me to do?"
        proposal = classification.tool_proposal
        record = None
        if proposal is not None:
            record = self._database.create_clarification(
                session_id=session_id,
                run_id=run_id,
                kind="tool_confirmation",
                question=question,
                proposed_tool_name=proposal.name,
                proposed_arguments=proposal.arguments,
            )

        await self._emit(
            "clarification_required",
            session_id,
            run_id,
            _clarification_payload(record)
            if record is not None
            else {
                "kind": "question",
                "question": question,
                "proposed_tool_name": None,
                "proposed_arguments": {},
                "status": "pending",
            },
        )
        await self._finish_without_tool(session_id, run_id, question)

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

        metadata = self._registry.get(classification.tool_proposal.name)
        selected_plan = self._tool_selected_plan(
            plan=plan,
            tool_name=classification.tool_proposal.name,
            risk_level=metadata.risk_level.value if metadata else None,
            requires_permission=False,
        )
        await self._execute_classified_tool(session_id, run_id, classification.tool_proposal, selected_plan)

    async def _handle_multi_step_task(
        self,
        session_id: str,
        run_id: str,
        user_message: str,
        classification: RequestClassification,
    ) -> None:
        sequence = list(classification.tool_sequence)
        if not sequence:
            await self._finish_without_tool(session_id, run_id, "I could not build a multi-step task from that request.")
            return
        is_browser_sequence = all(proposal.name.startswith("browser_") for proposal in sequence)

        def decide_next_action(state: TaskState) -> NextActionDecision:
            next_index = len(state.steps_taken)
            if next_index >= len(sequence):
                return NextActionDecision(
                    action_type="ANSWER",
                    decision_summary="All deterministic task-loop tool calls completed.",
                    tool_name=None,
                    arguments={"answer": "Completed the multi-step task."},
                )

            proposal = sequence[next_index]
            metadata = self._registry.get(proposal.name)
            return NextActionDecision(
                action_type="CALL_TOOL",
                decision_summary=f"Run deterministic task step {next_index + 1}: {proposal.name}.",
                tool_name=proposal.name,
                arguments=dict(proposal.arguments),
                risk_level=metadata.risk_level.value if metadata else None,
            )

        async def emit_task_event(event_type: str, event_session_id: str, event_run_id: str, data: dict[str, object]) -> None:
            if not is_browser_sequence:
                await self._emit(event_type, event_session_id, event_run_id, data)
                return
            if event_type == "task_loop_failed" and data.get("stop_reason") == "blocked_by_policy":
                await self._emit("browser_action_blocked", event_session_id, event_run_id, data)
            await self._emit(_browser_task_event_type(event_type), event_session_id, event_run_id, data)

        loop = TaskLoop(
            registry=self._registry,
            policy_engine=self._policy_engine,
            decide_next_action=decide_next_action,
            emit_event=emit_task_event,
        )
        state = await loop.run(session_id=session_id, run_id=run_id, user_goal=user_message)
        for step in state.steps_taken:
            result_content = step.observation.content if isinstance(step.observation.content, dict) else {"message": step.observation.content}
            output = result_content.get("output", {}) if isinstance(result_content, dict) else {}
            self._database.log_tool_call(
                session_id=session_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                status=step.status,
                result=output if isinstance(output, dict) else {"output": output},
                error=step.error,
            )
            if step.status == "success" and isinstance(output, dict):
                await self._emit_tool_artifact_if_any(
                    session_id,
                    run_id,
                    ToolCall(tool=step.tool_name, arguments=step.arguments, status=step.status, result=output, error=step.error),
                )
            if step.status != "success":
                self._database.create_reflection(
                    session_id=session_id,
                    run_id=run_id,
                    source_type="failed_tool",
                    note=f"Task-loop tool {step.tool_name} failed with error: {step.error or 'unknown error'}.",
                )

        if state.status == "completed":
            await self._finish_without_tool(session_id, run_id, "Completed the multi-step task.")
            return

        if state.status == "waiting_for_user":
            await self._finish_without_tool(session_id, run_id, "I need more information before continuing.")
            return

        await self._finish_without_tool(session_id, run_id, f"Multi-step task stopped with status: {state.status}.")

    async def _handle_local_context(
        self,
        session_id: str,
        run_id: str,
        user_message: str,
        classification: RequestClassification,
        plan: Plan,
    ) -> None:
        context_intent = classification.context_intent or resolve_context_intent(user_message)
        if context_intent is None and classification.tool_proposal is not None:
            context_intent = _context_intent_from_tool(classification.tool_proposal)

        if context_intent is None:
            await self._finish_without_tool(session_id, run_id, "I could not map that context request to a context tool.")
            return

        proposal = ToolProposal(context_intent.tool_name, dict(context_intent.arguments))
        await self._emit(
            "context_intent_detected",
            session_id,
            run_id,
            _context_intent_payload(context_intent),
        )

        context_window = None
        if proposal.name in {"get_selected_text", "get_context_window_info"}:
            context_window = self._context_window_data()
            await self._emit(
                "context_window_selected",
                session_id,
                run_id,
                {"window": context_window or {}, "tool_name": proposal.name},
            )

        metadata = self._registry.get(proposal.name)
        selected_plan = self._tool_selected_plan(
            plan=plan,
            tool_name=proposal.name,
            risk_level=metadata.risk_level.value if metadata else None,
            requires_permission=False,
        )
        context_result = await self._execute_context_tool(session_id, run_id, proposal, selected_plan)
        if context_result is None:
            return
        tool_call, capture_plan = context_result

        if tool_call.status != "success":
            await self._finish_without_tool(session_id, run_id, _context_capture_failure_message(context_intent, tool_call))
            return

        captured_text = _captured_context_text(context_intent, tool_call)
        if _context_requires_text(context_intent) and not captured_text.strip():
            await self._finish_without_tool(session_id, run_id, _context_capture_failure_message(context_intent, tool_call))
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
                "source": context_intent.source.value,
                "operation": context_intent.operation.value,
                "content_hash": _content_hash(captured_text or json.dumps(tool_call.result, sort_keys=True, default=str)),
            },
        )

        artifact = self._create_context_artifact(session_id, run_id, tool_call, context_intent)
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

        await self._answer_with_captured_context(user_message, context_intent, tool_call, session_id, run_id, capture_plan)

    async def _answer_with_captured_context(
        self,
        user_message: str,
        context_intent: ContextIntent,
        tool_call: ToolCall,
        session_id: str,
        run_id: str,
        plan: Plan,
    ) -> None:
        captured_text = _captured_context_text(context_intent, tool_call)
        if context_intent.operation == ContextOperation.report:
            completed_plan = update_plan_step(plan, 4, PlanStepStatus.completed)
            LOGGER.info(
                "context report answer completed | session=%s | run=%s | plan=%s",
                session_id,
                run_id,
                completed_plan.model_dump_json(),
            )
            await self._finish_without_tool(session_id, run_id, _report_context_message(context_intent, tool_call))
            return

        running_plan = update_plan_step(plan, 4, PlanStepStatus.running)
        messages = _captured_context_messages(user_message, context_intent, tool_call, captured_text)
        await self._emit(
            "llm_response_started",
            session_id,
            run_id,
            {
                "message_count": len(messages),
                "purpose": "local_context",
                "source": context_intent.source.value,
                "operation": context_intent.operation.value,
                "plan": running_plan.model_dump(mode="json"),
            },
        )
        try:
            assistant_message = await self._llm_client.complete(messages)
        except Exception as exc:
            LOGGER.exception("context llm completion failed | session=%s | run=%s", session_id, run_id)
            failed_plan = update_plan_step(running_plan, 4, PlanStepStatus.failed)
            await self._emit(
                "llm_response_finished",
                session_id,
                run_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "purpose": "local_context",
                    "source": context_intent.source.value,
                    "operation": context_intent.operation.value,
                    "plan": failed_plan.model_dump(mode="json"),
                },
            )
            await self._finish_without_tool(session_id, run_id, "I captured the context, but I could not get an AI answer right now.")
            return

        completed_plan = update_plan_step(running_plan, 4, PlanStepStatus.completed)
        await self._emit(
            "llm_response_finished",
            session_id,
            run_id,
            {
                "status": "success",
                "purpose": "local_context",
                "source": context_intent.source.value,
                "operation": context_intent.operation.value,
                "message": assistant_message,
                "plan": completed_plan.model_dump(mode="json"),
            },
        )
        final_message = assistant_message
        if not final_message or final_message == FALLBACK_MESSAGE:
            final_message = _fallback_context_answer(context_intent, tool_call)
        await self._finish_without_tool(session_id, run_id, final_message)

    async def _execute_context_tool(
        self,
        session_id: str,
        run_id: str,
        proposal,
        selected_plan: Plan,
    ) -> tuple[ToolCall, Plan] | None:
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

        running_plan = update_plan_step(selected_plan, 3, PlanStepStatus.running)
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
        final_plan = update_plan_step(running_plan, 3, final_status)
        if result.status != "success":
            final_plan = update_plan_step(final_plan, 4, PlanStepStatus.failed)
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
        return tool_call, final_plan

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
                    PlanStep(number=1, key="understand_question", title="Understand question", status=PlanStepStatus.completed),
                    PlanStep(
                        number=2,
                        key="generate_answer",
                        title="Generate answer",
                        status=PlanStepStatus.pending,
                        expected_observation="Assistant response text is generated without executing tools.",
                    ),
                ],
            )

        if classification.category == RequestCategory.no_action:
            return Plan(
                goal=f"No action: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_no_action", title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="confirm_no_action", title="Confirm no tool will run", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.ask_clarification:
            return Plan(
                goal=f"Clarify request: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_ambiguous_request", title="Understand ambiguous request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="ask_clarification", title="Ask clarification", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.realtime_info:
            return Plan(
                goal=f"Fetch real-time information: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_realtime_request", title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="select_tool", title="Select real-time tool", status=PlanStepStatus.pending),
                    PlanStep(number=3, key="execute_tool", title="Execute tool", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.local_context:
            return Plan(
                goal=f"Capture local context for: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_context_request", title="Understand context request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="select_context_tool", title="Select context tool", status=PlanStepStatus.pending),
                    PlanStep(
                        number=3,
                        key="capture_context",
                        title="Capture context",
                        status=PlanStepStatus.pending,
                        expected_observation="A local context tool returns captured text or window information.",
                    ),
                    PlanStep(
                        number=4,
                        key="answer_with_context",
                        title="Answer using captured context",
                        status=PlanStepStatus.pending,
                        expected_observation="Assistant response is grounded in the captured context artifact.",
                    ),
                ],
            )

        if classification.category == RequestCategory.multi_step:
            return Plan(
                goal=f"Run multi-step task: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_multi_step_task", title="Understand multi-step task", status=PlanStepStatus.completed),
                    PlanStep(
                        number=2,
                        key="run_task_loop",
                        title="Run task loop",
                        status=PlanStepStatus.pending,
                        expected_observation="Task loop emits public decisions and observations until it completes or stops.",
                    ),
                    PlanStep(number=3, key="finish_multi_step_task", title="Finish multi-step task", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.missing_tool:
            return Plan(
                goal=f"Handle missing capability: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_missing_capability", title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="propose_missing_tool", title="Propose missing tool", status=PlanStepStatus.pending),
                ],
            )

        if classification.category == RequestCategory.unsafe_or_blocked:
            return Plan(
                goal=f"Block unsafe request: {user_message}",
                steps=[
                    PlanStep(number=1, key="understand_unsafe_request", title="Understand request", status=PlanStepStatus.completed),
                    PlanStep(number=2, key="apply_safety_policy", title="Apply safety policy", status=PlanStepStatus.pending),
                ],
            )

        return self._planner.create_plan(user_message)

    async def _execute_pending_action(self, action: PendingAction) -> None:
        plan = Plan(
            goal=f"Run approved action: {action.tool_name}",
            steps=[
                PlanStep(number=1, key="understand_action", title="Understand action request", status=PlanStepStatus.completed),
                PlanStep(number=2, key="select_tool", title="Select tool", status=PlanStepStatus.completed, tool_name=action.tool_name),
                PlanStep(number=3, key="execute_tool", title="Execute tool", status=PlanStepStatus.pending, tool_name=action.tool_name),
            ],
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
        running_plan = update_plan_step(selected_plan, 3, PlanStepStatus.running)
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
        final_plan = update_plan_step(running_plan, 3, final_status)
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
        if result.status == "success":
            await self._emit_tool_artifact_if_any(session_id, run_id, tool_call)

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

    def _create_context_artifact(self, session_id: str, run_id: str, tool_call: ToolCall, context_intent: ContextIntent):
        artifact_type = _artifact_type_for_tool(tool_call.tool)
        if artifact_type is None:
            return None

        content_text = tool_call.result.get("text")
        content_text = str(content_text) if content_text is not None else None
        captured_at = str(tool_call.result.get("captured_at") or datetime.now(UTC).isoformat())
        context_window = tool_call.result.get("context_window") or tool_call.result.get("window") or {}
        content_for_hash = content_text if content_text is not None else json.dumps(tool_call.result, sort_keys=True, default=str)
        title = _artifact_title(tool_call)
        data = {
            "session_id": session_id,
            "run_id": run_id,
            "source": context_intent.source.value,
            "operation": context_intent.operation.value,
            "captured_at": captured_at,
            "context_window": context_window,
            "content_hash": _content_hash(content_for_hash),
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

    async def _emit_tool_artifact_if_any(self, session_id: str, run_id: str, tool_call: ToolCall) -> None:
        artifact = self._create_tool_artifact(session_id, run_id, tool_call)
        if artifact is None:
            return
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

    def _create_tool_artifact(self, session_id: str, run_id: str, tool_call: ToolCall):
        artifact_type = _artifact_type_for_tool(tool_call.tool)
        if artifact_type is None or not artifact_type.startswith("browser") and artifact_type != "form_draft":
            return None

        snapshot = tool_call.result.get("snapshot")
        title = _browser_artifact_title(tool_call, artifact_type)
        content_text = _browser_artifact_content(tool_call)
        data = {
            "session_id": session_id,
            "run_id": run_id,
            "tool": tool_call.tool,
            "arguments": tool_call.arguments,
            "result": tool_call.result,
            "risk_level": _browser_artifact_risk(tool_call),
            "snapshot": snapshot,
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
    def _tool_selected_plan(
        plan: Plan,
        tool_name: str,
        risk_level: str | None,
        requires_permission: bool,
    ) -> Plan:
        selected_plan = update_plan_steps(plan, {2: PlanStepStatus.completed})
        selected_plan = update_plan_step(
            selected_plan,
            2,
            tool_name=tool_name,
            risk_level=risk_level,
            requires_permission=requires_permission,
            expected_observation=f"{tool_name} is available and selected for controlled execution.",
        )
        return update_plan_step(
            selected_plan,
            3,
            tool_name=tool_name,
            risk_level=risk_level,
            requires_permission=requires_permission,
            expected_observation=f"{tool_name} returns a structured tool result.",
        )

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
    if tool_name == "browser_submit_form":
        return str(arguments.get("form_id", "")).strip() or "Unknown browser form"
    if tool_name == "browser_fill":
        return str(arguments.get("element_id", "")).strip() or "Unknown browser field"
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
    if tool_name == "browser_submit_form":
        return f"The assistant will submit browser form '{target}'. This may send data to the website."
    if tool_name == "browser_fill":
        return f"The assistant will fill browser field '{target}' as a draft. It will not submit the form."
    return f"The assistant will run '{tool_name}' with the shown content."


def _artifact_type_for_tool(tool_name: str) -> str | None:
    return {
        "get_clipboard_text": "clipboard_text",
        "get_selected_text": "selected_text",
        "get_context_window_info": "context_window_info",
        "get_foreground_window_info": "foreground_window_info",
        "browser_open": "browser_page_snapshot",
        "browser_read_page": "browser_page_snapshot",
        "browser_extract_links": "browser_links",
        "browser_extract_forms": "browser_forms",
        "browser_search": "browser_links",
        "browser_screenshot": "browser_screenshot",
        "browser_click": "browser_action_preview",
        "browser_fill": "form_draft",
        "browser_submit_form": "browser_action_preview",
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


def _browser_artifact_title(tool_call: ToolCall, artifact_type: str) -> str:
    title = str(tool_call.result.get("title") or "")
    url = str(tool_call.result.get("url") or "")
    if artifact_type == "browser_page_snapshot":
        return f"Page snapshot: {title or url or 'browser page'}"
    if artifact_type == "browser_links":
        return f"Links: {title or url or 'browser page'}"
    if artifact_type == "browser_forms":
        return f"Forms: {title or url or 'browser page'}"
    if artifact_type == "browser_screenshot":
        return f"Browser screenshot: {title or url or 'browser page'}"
    if artifact_type == "form_draft":
        return f"Form draft: {title or url or 'browser page'}"
    return f"Browser action: {tool_call.tool}"


def _browser_artifact_content(tool_call: ToolCall) -> str | None:
    if "text_preview" in tool_call.result:
        return str(tool_call.result.get("text_preview") or "")
    snapshot = tool_call.result.get("snapshot")
    if isinstance(snapshot, dict):
        return str(snapshot.get("text_preview") or "")
    if "links" in tool_call.result:
        return json.dumps(tool_call.result.get("links"), ensure_ascii=False, default=str)
    if "forms" in tool_call.result:
        return json.dumps(tool_call.result.get("forms"), ensure_ascii=False, default=str)
    return None


def _browser_artifact_risk(tool_call: ToolCall) -> str:
    form_draft = tool_call.result.get("form_draft")
    if isinstance(form_draft, dict):
        return str(form_draft.get("risk_level") or "MEDIUM_WRITE")
    if tool_call.tool == "browser_submit_form":
        return "HIGH_RISK"
    if tool_call.tool == "browser_fill":
        return "MEDIUM_WRITE"
    return "READ" if tool_call.tool in {"browser_read_page", "browser_extract_links", "browser_extract_forms", "browser_screenshot"} else "LOW_WRITE"


def _browser_task_event_type(event_type: str) -> str:
    return {
        "task_loop_started": "browser_task_started",
        "task_step_decided": "browser_step_decided",
        "task_observation_added": "browser_observation_added",
        "task_loop_completed": "browser_task_completed",
        "task_loop_failed": "browser_task_failed",
    }.get(event_type, event_type)


def _context_content_preview(tool_call: ToolCall) -> str:
    text = tool_call.result.get("text")
    if text is not None:
        return _short_text(str(text), limit=160)
    message = tool_call.result.get("message")
    if message is not None:
        return _short_text(str(message), limit=160)
    return ""


def _context_intent_payload(context_intent: ContextIntent) -> dict[str, object]:
    return {
        "source": context_intent.source.value,
        "operation": context_intent.operation.value,
        "tool_name": context_intent.tool_name,
        "arguments": context_intent.arguments,
        "confidence": context_intent.confidence,
        "reason": context_intent.reason,
    }


def _context_intent_from_tool(proposal: ToolProposal) -> ContextIntent | None:
    mapping = {
        "get_selected_text": ContextSource.selected_text,
        "get_clipboard_text": ContextSource.clipboard,
        "get_context_window_info": ContextSource.context_window,
        "get_foreground_window_info": ContextSource.foreground_window,
    }
    source = mapping.get(proposal.name)
    if source is None:
        return None
    return ContextIntent(
        source=source,
        operation=ContextOperation.unknown,
        tool_name=proposal.name,
        arguments=dict(proposal.arguments),
        reason="Context tool was selected by the classifier.",
        confidence=0.75,
    )


def _context_requires_text(context_intent: ContextIntent) -> bool:
    return context_intent.source in {ContextSource.selected_text, ContextSource.clipboard}


def _captured_context_text(context_intent: ContextIntent, tool_call: ToolCall) -> str:
    if context_intent.source in {ContextSource.selected_text, ContextSource.clipboard}:
        return str(tool_call.result.get("text") or "")
    return json.dumps(tool_call.result, ensure_ascii=False, sort_keys=True, default=str)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _context_capture_failure_message(context_intent: ContextIntent, tool_call: ToolCall) -> str:
    if context_intent.source == ContextSource.selected_text:
        detail = tool_call.error or "the selected text was empty"
        return f"I couldn't capture the highlighted or selected text. Please try again or copy it to the clipboard. ({detail})"
    if context_intent.source == ContextSource.clipboard:
        detail = tool_call.error or "the clipboard text was empty"
        return f"I couldn't read text from the clipboard. Please copy the text and try again. ({detail})"
    return tool_call.error or f"{tool_call.tool} failed."


def _report_context_message(context_intent: ContextIntent, tool_call: ToolCall) -> str:
    text = _captured_context_text(context_intent, tool_call).strip()
    if context_intent.source == ContextSource.selected_text:
        return f"You highlighted: {text}"
    if context_intent.source == ContextSource.clipboard:
        return f"Clipboard contains: {text}"
    message = tool_call.result.get("message")
    if message:
        return str(message)
    return json.dumps(tool_call.result, ensure_ascii=False, indent=2, default=str)


def _fallback_context_answer(context_intent: ContextIntent, tool_call: ToolCall) -> str:
    if context_intent.operation == ContextOperation.report:
        return _report_context_message(context_intent, tool_call)
    if context_intent.source in {ContextSource.selected_text, ContextSource.clipboard}:
        return (
            "I captured the requested context, but I couldn't generate a polished answer. "
            f"Captured text: {_captured_context_text(context_intent, tool_call).strip()}"
        )
    return _report_context_message(context_intent, tool_call)


def _captured_context_messages(
    user_message: str,
    context_intent: ContextIntent,
    tool_call: ToolCall,
    captured_text: str,
) -> list[dict[str, object]]:
    context_payload = captured_text if context_intent.source in {ContextSource.selected_text, ContextSource.clipboard} else json.dumps(
        tool_call.result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return [
        {
            "role": "system",
            "content": (
                "You are Vix, a Windows desktop assistant. The backend has already captured local context using a tool. "
                "Treat the captured context as authoritative. Do not say you cannot see the selection, clipboard, or screen "
                "if captured context is provided. If the captured context is insufficient, say exactly what is missing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user request: {user_message}\n"
                f"Captured context source: {context_intent.source.value}\n"
                f"Captured context operation: {context_intent.operation.value}\n"
                f"Captured context:\n\"\"\"\n{context_payload}\n\"\"\"\n\n"
                "Task: If the user asked for the answer to a highlighted/selected question, answer that captured question directly. "
                "If the user asked what was highlighted, report the captured text. If the user asked for meaning, explain the captured text. "
                "Do not invent page content that is not in captured context. For Stage 4, if selected text is only a question, "
                "answering the question using general knowledge is allowed. If the user asks for an answer that appears elsewhere on the page "
                "and the selected text does not contain enough information, say that page-reading, screenshot, or browser context is needed."
            ),
        },
    ]


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


def _clarification_reply_kind(message: str) -> str | None:
    normalized = canonicalize_user_text(message).strip(" .,!?:;")
    accept_replies = {
        "yes",
        "y",
        "yeah",
        "sure",
        "ok",
        "okay",
        "open it",
        "do it",
    }
    reject_replies = {
        "no",
        "n",
        "nope",
        "cancel",
        "stop",
        "don't",
        "dont",
        "do not",
    }
    if normalized in accept_replies:
        return "accept"
    if normalized in reject_replies:
        return "reject"
    return None


def _clarification_arguments(record: ClarificationRecord) -> dict[str, object]:
    try:
        parsed = json.loads(record.proposed_arguments_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clarification_payload(record: ClarificationRecord) -> dict[str, object]:
    return {
        "clarification_id": record.id,
        "kind": record.kind,
        "question": record.question,
        "proposed_tool_name": record.proposed_tool_name,
        "proposed_arguments": _clarification_arguments(record),
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
        "decided_at": record.decided_at.isoformat() if record.decided_at is not None else None,
    }
