from fastapi import APIRouter, Request

from app.assistant.orchestrator import Orchestrator
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    orchestrator: Orchestrator = request.app.state.orchestrator
    return await orchestrator.handle_chat(payload)
