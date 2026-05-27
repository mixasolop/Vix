from abc import ABC, abstractmethod

from app.schemas.tools import ToolDefinition, ToolResult


class AssistantTool(ABC):
    definition: ToolDefinition

    @abstractmethod
    async def run(self, arguments: dict[str, object]) -> ToolResult:
        raise NotImplementedError
