import json
from typing import Protocol

from app.assistant.missing_tool_proposer import propose_missing_tool
from app.assistant.planner import FALLBACK_MESSAGE
from app.schemas.plans import AssistantPlan, PlanStep, PlanStepStatus
from app.schemas.proposed_tools import ALLOWED_PROPOSED_TOOL_RISK_LEVELS, ProposedToolDraft
from app.schemas.tools import ToolDefinition

OPENAI_PROPOSAL_SYSTEM_PROMPT = """
You generate proposed tool specifications for a safe desktop assistant.

Policy:
- Return only one ProposedToolDraft object.
- Do not generate runnable code.
- Do not execute tools.
- Do not modify files.
- Do not approve proposals.
- The proposed tool is for human developer review only.
- Use risk_level exactly as one of: READ, LOW_WRITE, MEDIUM_WRITE, HIGH_RISK.
""".strip()


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, object]]) -> str:
        """Return a normal assistant reply. Stage 1 does not use this to execute tools."""

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        """Return a validated plan shape for future LLM tool planning."""

    async def propose_tool_spec(self, user_message: str, existing_tools: list[ToolDefinition]) -> ProposedToolDraft | None:
        """Return a proposed tool spec. Stage 2 stores it for developer review only."""


class DeterministicLLMClient:
    async def complete(self, messages: list[dict[str, object]]) -> str:
        return FALLBACK_MESSAGE

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        return AssistantPlan(
            goal=f"Respond to: {user_text.strip()}" if user_text.strip() else "Respond to empty request",
            steps=[
                PlanStep(number=1, title="Produce normal assistant reply", status=PlanStepStatus.pending),
            ],
        )

    async def propose_tool_spec(self, user_message: str, existing_tools: list[ToolDefinition]) -> ProposedToolDraft | None:
        return propose_missing_tool(user_message)


class OpenAILLMClient:
    def __init__(self, api_key: str, model: str = "gpt-5.4-mini", client: object | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    async def complete(self, messages: list[dict[str, object]]) -> str:
        return FALLBACK_MESSAGE

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        return await DeterministicLLMClient().create_plan(user_text, tools)

    async def propose_tool_spec(self, user_message: str, existing_tools: list[ToolDefinition]) -> ProposedToolDraft | None:
        response = await self._get_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": OPENAI_PROPOSAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_message": user_message,
                            "existing_tools": [
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "status": tool.status.value,
                                    "risk_level": tool.risk_level.value,
                                    "input_schema": tool.input_schema,
                                    "output_schema": tool.output_schema,
                                }
                                for tool in existing_tools
                            ],
                        },
                        sort_keys=True,
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "proposed_tool_draft",
                    "strict": False,
                    "schema": _proposed_tool_draft_schema(),
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return ProposedToolDraft.model_validate_json(content)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client


def _proposed_tool_draft_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "reason": {"type": "string"},
            "risk_level": {"type": "string", "enum": sorted(ALLOWED_PROPOSED_TOOL_RISK_LEVELS)},
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        "required": ["name", "description", "reason", "risk_level", "input_schema", "output_schema"],
        "additionalProperties": False,
    }
