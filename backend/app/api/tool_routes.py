from fastapi import APIRouter, Request

from app.schemas.tools import ToolListResponse
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(request: Request) -> ToolListResponse:
    registry: ToolRegistry = request.app.state.registry
    return ToolListResponse(tools=registry.list_tools())
