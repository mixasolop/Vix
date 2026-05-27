from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI

from app.api.chat_routes import router as chat_router
from app.api.permission_routes import router as permission_router
from app.api.proposed_tool_routes import router as proposed_tool_router
from app.api.tool_routes import router as tool_router
from app.assistant.llm_client import DeterministicLLMClient, LLMClient, OpenAILLMClient
from app.assistant.orchestrator import Orchestrator
from app.assistant.policy import PermissionManager, PolicyEngine
from app.config import AppConfig, load_config
from app.db.database import Database
from app.events.event_bus import EventBus
from app.logging_config import configure_logging
from app.schemas.ai import AIStatusResponse
from app.tools.registry import build_default_registry
from app.tools.registry import ToolRegistry
from app.ws.event_stream import router as event_stream_router

configure_logging()
LOGGER = logging.getLogger("app.main")


def create_app(
    database_path: str | Path | None = None,
    registry: ToolRegistry | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    config = load_config()
    database = Database(database_path or config.database_path)
    event_bus = EventBus()
    registry = registry or build_default_registry()
    llm_client = llm_client or _build_llm_client(config)
    permission_manager = PermissionManager(database)
    policy_engine = PolicyEngine()
    orchestrator = Orchestrator(
        registry=registry,
        database=database,
        permission_manager=permission_manager,
        event_bus=event_bus,
        policy_engine=policy_engine,
        llm_client=llm_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        LOGGER.info("backend startup beginning")
        app.state.database.initialize()
        app.state.database.upsert_tools(app.state.registry.list_tools())
        tools = app.state.registry.list_tools()
        implemented_count = sum(1 for tool in tools if tool.status.value == "implemented")
        planned_count = sum(1 for tool in tools if tool.status.value == "planned")
        disabled_count = sum(1 for tool in tools if tool.status.value == "disabled")
        LOGGER.info(
            "backend ready | database=%s | tools=%s implemented=%s planned=%s disabled=%s | health=http://127.0.0.1:8000/health | ws=ws://127.0.0.1:8000/ws/events",
            app.state.database.path,
            len(tools),
            implemented_count,
            planned_count,
            disabled_count,
        )
        try:
            yield
        finally:
            LOGGER.info("backend shutdown beginning")
            app.state.database.close()
            LOGGER.info("backend shutdown complete")

    app = FastAPI(
        title="Desktop Assistant Backend",
        version="0.1.0",
        description="Stage 2 safe tool proposal and review runtime.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.database = database
    app.state.event_bus = event_bus
    app.state.registry = registry
    app.state.permission_manager = permission_manager
    app.state.policy_engine = policy_engine
    app.state.orchestrator = orchestrator

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "desktop-assistant-backend",
            "stage": "2.0",
        }

    @app.get("/ai/status", response_model=AIStatusResponse)
    async def ai_status() -> AIStatusResponse:
        return await _get_ai_status(app.state.config, app.state.orchestrator.llm_client)

    app.include_router(chat_router)
    app.include_router(tool_router)
    app.include_router(permission_router)
    app.include_router(proposed_tool_router)
    app.include_router(event_stream_router)
    return app


def _build_llm_client(config: AppConfig) -> LLMClient:
    if not config.ai_proposals_enabled:
        return DeterministicLLMClient()

    if config.ai_provider.lower() != "openai":
        LOGGER.warning("AI proposals disabled because unsupported AI_PROVIDER=%s", config.ai_provider)
        return DeterministicLLMClient()

    if not config.openai_api_key:
        LOGGER.warning("AI proposals disabled because OPENAI_API_KEY is not configured")
        return DeterministicLLMClient()

    return OpenAILLMClient(api_key=config.openai_api_key, model=config.ai_proposal_model)


async def _get_ai_status(config: AppConfig, llm_client: LLMClient) -> AIStatusResponse:
    api_key_configured = bool(config.openai_api_key)
    if not config.ai_proposals_enabled:
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            proposals_enabled=False,
            api_key_configured=api_key_configured,
            connected=False,
            status="disabled",
            detail="AI proposals are disabled. Set AI_PROPOSALS_ENABLED=true to verify and use OpenAI proposals.",
        )

    if config.ai_provider.lower() != "openai":
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            proposals_enabled=True,
            api_key_configured=api_key_configured,
            connected=False,
            status="unsupported_provider",
            detail=f"Unsupported AI_PROVIDER '{config.ai_provider}'. Only OpenAI is implemented.",
        )

    if not api_key_configured:
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            proposals_enabled=True,
            api_key_configured=False,
            connected=False,
            status="missing_api_key",
            detail="OPENAI_API_KEY is not configured.",
        )

    connected, detail = await llm_client.verify_connection()
    return AIStatusResponse(
        provider=config.ai_provider,
        model=config.ai_proposal_model,
        proposals_enabled=True,
        api_key_configured=True,
        connected=connected,
        status="connected" if connected else "verification_failed",
        detail=detail,
    )


app = create_app()
