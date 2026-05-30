from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI

from app.api.browser_routes import router as browser_router
from app.api.chat_routes import router as chat_router
from app.api.context_routes import router as context_router
from app.api.permission_routes import router as permission_router
from app.api.proposed_tool_routes import router as proposed_tool_router
from app.api.tool_routes import router as tool_router
from app.assistant.llm_client import DeterministicLLMClient, LLMClient, OpenAILLMClient, redact_secrets
from app.assistant.orchestrator import Orchestrator
from app.assistant.policy import PermissionManager, PolicyEngine
from app.assistant.semantic_router import SemanticRouter
from app.browser.session_manager import BrowserSessionManager
from app.config import AppConfig, load_config, load_config_from_file
from app.context.window_tracker import WindowTracker
from app.db.database import Database
from app.events.event_bus import EventBus
from app.logging_config import configure_logging
from app.schemas.events import AssistantEvent
from app.schemas.ai import AIStatusResponse
from app.schemas.window_context import WindowInfo
from app.tools.registry import build_default_registry
from app.tools.registry import ToolRegistry
from app.ws.event_stream import router as event_stream_router

configure_logging()
LOGGER = logging.getLogger("app.main")


def create_app(
    database_path: str | Path | None = None,
    registry: ToolRegistry | None = None,
    llm_client: LLMClient | None = None,
    config: AppConfig | None = None,
    reload_config_from_file: bool | None = None,
    context_tracker: WindowTracker | None = None,
    browser_manager: BrowserSessionManager | None = None,
    start_context_tracker: bool = False,
    semantic_router: SemanticRouter | None = None,
) -> FastAPI:
    should_reload_config = config is None if reload_config_from_file is None else reload_config_from_file
    user_supplied_llm_client = llm_client is not None
    config = config or load_config()
    database = Database(database_path or config.database_path)
    event_bus = EventBus()
    context_tracker = context_tracker or WindowTracker()
    browser_manager = browser_manager or BrowserSessionManager()
    registry = registry or build_default_registry(context_tracker, browser_manager=browser_manager)
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
        context_tracker=context_tracker,
        semantic_router=semantic_router,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        LOGGER.info("backend startup beginning")
        app.state.database.initialize()
        app.state.database.upsert_tools(app.state.registry.list_tools())
        loop = asyncio.get_running_loop()

        if app.state.start_context_tracker:
            def on_context_window_updated(window: WindowInfo) -> None:
                loop.call_soon_threadsafe(lambda: asyncio.create_task(_emit_context_window_updated(app, window)))

            app.state.context_tracker.set_context_updated_callback(on_context_window_updated)
            app.state.context_tracker.start()
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
            if app.state.start_context_tracker:
                app.state.context_tracker.stop()
            await app.state.browser_manager.stop()
            app.state.database.close()
            LOGGER.info("backend shutdown complete")

    app = FastAPI(
        title="Desktop Assistant Backend",
        version="0.2.0",
        description="Stage 5 controlled browser, local context, general answer, weather, and safe tool proposal runtime.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.database = database
    app.state.event_bus = event_bus
    app.state.registry = registry
    app.state.permission_manager = permission_manager
    app.state.policy_engine = policy_engine
    app.state.orchestrator = orchestrator
    app.state.context_tracker = context_tracker
    app.state.browser_manager = browser_manager
    app.state.start_context_tracker = start_context_tracker
    app.state.reload_config_from_file = should_reload_config
    app.state.user_supplied_llm_client = user_supplied_llm_client
    app.state.ai_config_signature = _ai_config_signature(config)

    def refresh_runtime_config() -> AppConfig:
        if not app.state.reload_config_from_file:
            return app.state.config

        refreshed_config = load_config_from_file(app.state.config.config_file_path)
        if refreshed_config == app.state.config:
            return app.state.config

        previous_signature = app.state.ai_config_signature
        next_signature = _ai_config_signature(refreshed_config)
        app.state.config = refreshed_config
        LOGGER.info(
            "runtime config reloaded | file=%s | ai_provider=%s | ai_model=%s | ai_general_answers_enabled=%s | ai_proposals_enabled=%s | api_key_configured=%s | api_key_fingerprint=%s",
            refreshed_config.config_file_path,
            refreshed_config.ai_provider,
            refreshed_config.ai_proposal_model,
            refreshed_config.ai_general_answers_enabled,
            refreshed_config.ai_proposals_enabled,
            bool(refreshed_config.openai_api_key),
            _api_key_fingerprint(refreshed_config.openai_api_key),
        )

        if previous_signature != next_signature and not app.state.user_supplied_llm_client:
            app.state.orchestrator.set_llm_client(_build_llm_client(refreshed_config))
            app.state.ai_config_signature = next_signature
            LOGGER.info(
                "llm client refreshed | ai_provider=%s | ai_model=%s | ai_general_answers_enabled=%s | ai_proposals_enabled=%s",
                refreshed_config.ai_provider,
                refreshed_config.ai_proposal_model,
                refreshed_config.ai_general_answers_enabled,
                refreshed_config.ai_proposals_enabled,
            )
        return app.state.config

    app.state.refresh_runtime_config = refresh_runtime_config

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "desktop-assistant-backend",
            "stage": "5.0",
        }

    @app.get("/ai/status", response_model=AIStatusResponse)
    async def ai_status() -> AIStatusResponse:
        app.state.refresh_runtime_config()
        return await _get_ai_status(app.state.config, app.state.orchestrator.llm_client)

    app.include_router(chat_router)
    app.include_router(browser_router)
    app.include_router(context_router)
    app.include_router(tool_router)
    app.include_router(permission_router)
    app.include_router(proposed_tool_router)
    app.include_router(event_stream_router)
    return app


async def _emit_context_window_updated(app: FastAPI, window: WindowInfo) -> None:
    data = {
        "window": window.model_dump(mode="json"),
        "title": window.title,
        "process_name": window.process_name,
        "is_vix": window.is_vix,
    }
    event = AssistantEvent(type="context_window_updated", data=data)
    app.state.database.log_event(event)
    await app.state.event_bus.publish(event)


def _build_llm_client(config: AppConfig) -> LLMClient:
    if not config.ai_general_answers_enabled and not config.ai_proposals_enabled:
        return DeterministicLLMClient()

    if config.ai_provider.lower() != "openai":
        LOGGER.warning("AI disabled because unsupported AI_PROVIDER=%s", config.ai_provider)
        return DeterministicLLMClient()

    if not config.openai_api_key:
        LOGGER.warning("AI disabled because OPENAI_API_KEY is not configured")
        return DeterministicLLMClient()

    return OpenAILLMClient(
        api_key=config.openai_api_key,
        model=config.ai_proposal_model,
        tool_proposals_enabled=config.ai_proposals_enabled,
    )


def _ai_config_signature(config: AppConfig) -> tuple[str, str, bool, bool, str | None]:
    return (
        config.ai_provider,
        config.ai_proposal_model,
        config.ai_general_answers_enabled,
        config.ai_proposals_enabled,
        config.openai_api_key,
    )


async def _get_ai_status(config: AppConfig, llm_client: LLMClient) -> AIStatusResponse:
    api_key_configured = bool(config.openai_api_key)
    api_key_fingerprint = _api_key_fingerprint(config.openai_api_key)
    if not config.ai_general_answers_enabled and not config.ai_proposals_enabled:
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            config_file_path=str(config.config_file_path),
            general_answers_enabled=False,
            proposals_enabled=False,
            api_key_configured=api_key_configured,
            api_key_fingerprint=api_key_fingerprint,
            model_reachable=False,
            connected=False,
            status="disabled",
            general_answers_status="disabled",
            tool_proposals_status="disabled",
            api_key_status="configured" if api_key_configured else "missing",
            model_status="not_checked",
            detail="AI is disabled. Set AI_GENERAL_ANSWERS_ENABLED=true or AI_PROPOSALS_ENABLED=true in backend/.env.",
            tool_execution_mode="deterministic",
        )

    general_answers_status = "enabled" if config.ai_general_answers_enabled else "disabled"
    tool_proposals_status = "enabled" if config.ai_proposals_enabled else "disabled"

    if config.ai_provider.lower() != "openai":
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            config_file_path=str(config.config_file_path),
            general_answers_enabled=config.ai_general_answers_enabled,
            proposals_enabled=config.ai_proposals_enabled,
            api_key_configured=api_key_configured,
            api_key_fingerprint=api_key_fingerprint,
            model_reachable=False,
            connected=False,
            status="unsupported_provider",
            general_answers_status=general_answers_status,
            tool_proposals_status=tool_proposals_status,
            api_key_status="configured" if api_key_configured else "missing",
            model_status="not_checked",
            detail=f"Unsupported AI_PROVIDER '{config.ai_provider}'. Only OpenAI is implemented.",
            tool_execution_mode="deterministic",
        )

    if not api_key_configured:
        return AIStatusResponse(
            provider=config.ai_provider,
            model=config.ai_proposal_model,
            config_file_path=str(config.config_file_path),
            general_answers_enabled=config.ai_general_answers_enabled,
            proposals_enabled=config.ai_proposals_enabled,
            api_key_configured=False,
            api_key_fingerprint=None,
            model_reachable=False,
            connected=False,
            status="missing_api_key",
            general_answers_status="missing_api_key" if config.ai_general_answers_enabled else "disabled",
            tool_proposals_status="missing_api_key" if config.ai_proposals_enabled else "disabled",
            api_key_status="missing",
            model_status="not_checked",
            detail="OPENAI_API_KEY is not configured in backend/.env.",
            tool_execution_mode="deterministic",
        )

    try:
        connected, detail = await asyncio.wait_for(llm_client.verify_connection(), timeout=8)
        detail = redact_secrets(detail)
    except TimeoutError:
        connected = False
        detail = "OpenAI model verification timed out after 8 seconds."
    model_status = "reachable" if connected else "unreachable"
    return AIStatusResponse(
        provider=config.ai_provider,
        model=config.ai_proposal_model,
        config_file_path=str(config.config_file_path),
        general_answers_enabled=config.ai_general_answers_enabled,
        proposals_enabled=config.ai_proposals_enabled,
        api_key_configured=True,
        api_key_fingerprint=api_key_fingerprint,
        model_reachable=connected,
        connected=connected,
        status="connected" if connected else "verification_failed",
        general_answers_status=("enabled" if connected else "model_unreachable") if config.ai_general_answers_enabled else "disabled",
        tool_proposals_status=("enabled" if connected else "model_unreachable") if config.ai_proposals_enabled else "disabled",
        api_key_status="configured",
        model_status=model_status,
        detail=detail,
        tool_execution_mode="deterministic",
    )


def _api_key_fingerprint(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


app = create_app(start_context_tracker=True)
