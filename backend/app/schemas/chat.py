from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class ChatAcceptedResponse(BaseModel):
    accepted: bool
    conversation_id: str
    run_id: str
