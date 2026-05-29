from fastapi import APIRouter, Request

from app.assistant.orchestrator import Orchestrator
from app.schemas.chat import ChatAcceptedResponse, ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatAcceptedResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatAcceptedResponse:
    refresh_runtime_config = getattr(request.app.state, "refresh_runtime_config", None)
    if refresh_runtime_config is not None:
        refresh_runtime_config()

    orchestrator: Orchestrator = request.app.state.orchestrator
    return await orchestrator.start_chat(payload)
