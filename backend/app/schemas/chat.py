from pydantic import BaseModel, Field

from app.schemas.plans import Plan
from app.schemas.tools import PermissionRequest, ToolCall


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    plan: Plan
    tool_calls: list[ToolCall]
    permissions: list[PermissionRequest]
