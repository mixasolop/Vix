from fastapi import APIRouter, HTTPException, Request

from app.db.database import Database
from app.events.event_bus import EventBus
from app.schemas.events import AssistantEvent
from app.schemas.proposed_tools import (
    CreateProposedToolRequest,
    ProposedTool,
    ProposedToolListResponse,
    ProposedToolStatus,
)

router = APIRouter(prefix="/proposed-tools", tags=["proposed-tools"])


@router.get("", response_model=ProposedToolListResponse)
async def list_proposed_tools(request: Request) -> ProposedToolListResponse:
    database: Database = request.app.state.database
    return ProposedToolListResponse(tools=database.list_proposed_tools())


@router.post("", response_model=ProposedTool)
async def create_proposed_tool(payload: CreateProposedToolRequest, request: Request) -> ProposedTool:
    database: Database = request.app.state.database
    proposed_tool = database.create_proposed_tool(payload)
    await _emit_status_event(request, "proposed_tool_created", proposed_tool)
    return proposed_tool


@router.post("/{tool_id}/approve", response_model=ProposedTool)
async def approve_proposed_tool(tool_id: str, request: Request) -> ProposedTool:
    proposed_tool = await _update_status(request, tool_id, ProposedToolStatus.approved)
    await _emit_status_event(request, "proposed_tool_approved", proposed_tool)
    return proposed_tool


@router.post("/{tool_id}/reject", response_model=ProposedTool)
async def reject_proposed_tool(tool_id: str, request: Request) -> ProposedTool:
    proposed_tool = await _update_status(request, tool_id, ProposedToolStatus.rejected)
    await _emit_status_event(request, "proposed_tool_rejected", proposed_tool)
    return proposed_tool


@router.post("/{tool_id}/needs-changes", response_model=ProposedTool)
async def mark_proposed_tool_needs_changes(tool_id: str, request: Request) -> ProposedTool:
    proposed_tool = await _update_status(request, tool_id, ProposedToolStatus.needs_changes)
    await _emit_status_event(request, "proposed_tool_needs_changes", proposed_tool)
    return proposed_tool


async def _update_status(request: Request, tool_id: str, status: ProposedToolStatus) -> ProposedTool:
    database: Database = request.app.state.database
    proposed_tool = database.update_proposed_tool_status(tool_id, status)
    if proposed_tool is None:
        raise HTTPException(status_code=404, detail=f"Proposed tool not found: {tool_id}")
    return proposed_tool


async def _emit_status_event(request: Request, event_type: str, proposed_tool: ProposedTool) -> None:
    event_bus: EventBus = request.app.state.event_bus
    event = AssistantEvent(
        type=event_type,
        data={
            "tool_id": proposed_tool.id,
            "name": proposed_tool.name,
            "reason": proposed_tool.reason,
            "risk_level": proposed_tool.risk_level,
            "status": proposed_tool.status.value,
            "tool": proposed_tool.model_dump(mode="json"),
        },
    )
    request.app.state.database.log_event(event)
    await event_bus.publish(event)
