from fastapi import APIRouter, Request

from app.assistant.orchestrator import Orchestrator
from app.schemas.chat import ChatAcceptedResponse, ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatAcceptedResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatAcceptedResponse:
    orchestrator: Orchestrator = request.app.state.orchestrator
    return await orchestrator.start_chat(payload)
