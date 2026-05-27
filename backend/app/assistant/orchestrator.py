from uuid import uuid4

from app.assistant.planner import Planner
from app.assistant.policy import PermissionManager
from app.db.database import Database
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.events import AssistantEvent
from app.schemas.tools import ToolCall
from app.tools.registry import ToolRegistry
from app.tools.system_tools import fake_launch_app


class Orchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        database: Database,
        permission_manager: PermissionManager,
        planner: Planner | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self._permission_manager = permission_manager
        self._planner = planner or Planner()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or f"local-{uuid4()}"
        self._log_event(
            "chat.message.received",
            {"conversation_id": conversation_id, "message": request.message},
        )

        plan = self._planner.create_plan(request.message)
        self._log_event("plan.created", {"conversation_id": conversation_id, "goal": plan.goal})

        app_name = self._select_fake_app_name(request.message)
        result = fake_launch_app(app_name)
        tool_call = ToolCall(
            tool="launch_app",
            arguments={"app_name": app_name},
            status=result.status,
            result=result.output,
        )
        self._log_event(
            "tool.finished",
            {
                "conversation_id": conversation_id,
                "tool": tool_call.tool,
                "status": tool_call.status,
            },
        )

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=f"Fake result: {app_name} would be opened.",
            plan=plan,
            tool_calls=[tool_call],
            permissions=[],
        )

    def _log_event(self, event_type: str, payload: dict[str, object]) -> None:
        self._database.log_event(AssistantEvent(type=event_type, payload=payload))

    @staticmethod
    def _select_fake_app_name(user_message: str) -> str:
        if "notepad" in user_message.lower():
            return "notepad"
        return "notepad"
