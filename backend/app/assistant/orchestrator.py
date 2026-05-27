from uuid import uuid4

from app.assistant.planner import Planner
from app.assistant.policy import PermissionManager
from app.db.database import Database
from app.events.event_bus import EventBus
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.events import AssistantEvent
from app.schemas.tools import ConfirmationPolicy, PermissionRequest, ToolCall
from app.tools.registry import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        database: Database,
        permission_manager: PermissionManager,
        event_bus: EventBus,
        planner: Planner | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self._permission_manager = permission_manager
        self._event_bus = event_bus
        self._planner = planner or Planner()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or f"local-{uuid4()}"
        self._database.ensure_session(conversation_id)
        self._database.log_message(conversation_id, "user", request.message)
        await self._emit(
            "chat.message.received",
            {"conversation_id": conversation_id, "role": "user", "message": request.message},
        )

        plan = self._planner.create_plan(request.message)
        await self._emit(
            "plan.created",
            {"conversation_id": conversation_id, "plan": plan.model_dump(mode="json")},
        )

        proposal = self._planner.propose_tool_call(request.message)
        if proposal is None:
            assistant_message = "I could not map that request to an implemented Stage 1 tool yet."
            self._database.log_message(conversation_id, "assistant", assistant_message)
            await self._emit(
                "assistant.message.created",
                {"conversation_id": conversation_id, "message": assistant_message},
            )
            return ChatResponse(
                conversation_id=conversation_id,
                assistant_message=assistant_message,
                plan=plan,
                tool_calls=[],
                permissions=[],
            )

        metadata = self._registry.get(proposal.name)
        if metadata is None:
            assistant_message = f"Tool is not registered: {proposal.name}."
            self._database.log_message(conversation_id, "assistant", assistant_message)
            await self._emit("tool.failed", {"conversation_id": conversation_id, "tool": proposal.name, "error": assistant_message})
            return ChatResponse(
                conversation_id=conversation_id,
                assistant_message=assistant_message,
                plan=plan,
                tool_calls=[],
                permissions=[],
            )

        if metadata.confirmation_policy != ConfirmationPolicy.none:
            permission_id = str(uuid4())
            permission = PermissionRequest(
                permission_id=permission_id,
                tool=proposal.name,
                reason=f"{proposal.name} requires confirmation before execution.",
                arguments=proposal.arguments,
            )
            self._database.create_permission(
                permission_id=permission_id,
                session_id=conversation_id,
                tool_name=proposal.name,
                reason=permission.reason,
                arguments=proposal.arguments,
            )
            await self._emit(
                "permission.required",
                {
                    "conversation_id": conversation_id,
                    "permission_id": permission_id,
                    "tool": proposal.name,
                    "arguments": proposal.arguments,
                    "reason": permission.reason,
                },
            )
            assistant_message = f"Permission is required before running {proposal.name}."
            self._database.log_message(conversation_id, "assistant", assistant_message)
            return ChatResponse(
                conversation_id=conversation_id,
                assistant_message=assistant_message,
                plan=plan,
                tool_calls=[],
                permissions=[permission],
            )

        await self._emit(
            "tool.started",
            {"conversation_id": conversation_id, "tool": proposal.name, "arguments": proposal.arguments},
        )
        result = await self._registry.execute(proposal.name, proposal.arguments)
        self._database.log_tool_call(
            session_id=conversation_id,
            tool_name=proposal.name,
            arguments=proposal.arguments,
            status=result.status,
            result=result.output,
            error=result.error,
        )

        tool_call = ToolCall(
            tool=proposal.name,
            arguments=proposal.arguments,
            status=result.status,
            result=result.output,
            error=result.error,
        )
        await self._emit(
            "tool.finished",
            {
                "conversation_id": conversation_id,
                "tool": tool_call.tool,
                "arguments": tool_call.arguments,
                "status": tool_call.status,
                "result": tool_call.result,
                "error": tool_call.error,
            },
        )

        assistant_message = self._assistant_message_for_tool_result(tool_call)
        self._database.log_message(conversation_id, "assistant", assistant_message)
        await self._emit(
            "assistant.message.created",
            {"conversation_id": conversation_id, "message": assistant_message},
        )

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            plan=plan,
            tool_calls=[tool_call],
            permissions=[],
        )

    async def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        event = AssistantEvent(type=event_type, payload=payload)
        self._database.log_event(event)
        await self._event_bus.publish(event)

    @staticmethod
    def _assistant_message_for_tool_result(tool_call: ToolCall) -> str:
        if tool_call.status == "success":
            message = tool_call.result.get("message")
            return str(message) if message else f"{tool_call.tool} completed."
        return tool_call.error or f"{tool_call.tool} failed."
