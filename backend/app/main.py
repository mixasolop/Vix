from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat_routes import router as chat_router
from app.api.permission_routes import router as permission_router
from app.api.tool_routes import router as tool_router
from app.assistant.orchestrator import Orchestrator
from app.assistant.policy import PermissionManager
from app.config import load_config
from app.db.database import Database
from app.events.event_bus import EventBus
from app.tools.registry import build_default_registry
from app.tools.registry import ToolRegistry
from app.ws.event_stream import router as event_stream_router


def create_app(database_path: str | Path | None = None, registry: ToolRegistry | None = None) -> FastAPI:
    config = load_config()
    database = Database(database_path or config.database_path)
    event_bus = EventBus()
    registry = registry or build_default_registry()
    permission_manager = PermissionManager(database)
    orchestrator = Orchestrator(
        registry=registry,
        database=database,
        permission_manager=permission_manager,
        event_bus=event_bus,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.database.initialize()
        yield
        app.state.database.close()

    app = FastAPI(
        title="Desktop Assistant Backend",
        version="0.1.0",
        description="Stage 1.2 skeleton backend with fake planning and tool results.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.database = database
    app.state.event_bus = event_bus
    app.state.registry = registry
    app.state.permission_manager = permission_manager
    app.state.orchestrator = orchestrator

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "desktop-assistant-backend",
            "stage": "1.2",
        }

    app.include_router(chat_router)
    app.include_router(tool_router)
    app.include_router(permission_router)
    app.include_router(event_stream_router)
    return app


app = create_app()
