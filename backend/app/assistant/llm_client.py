import json
import logging
import re
from typing import Protocol

from app.assistant.missing_tool_proposer import propose_missing_tool
from app.assistant.planner import FALLBACK_MESSAGE
from app.schemas.plans import AssistantPlan, PlanStep, PlanStepStatus
from app.schemas.proposed_tools import ALLOWED_PROPOSED_TOOL_RISK_LEVELS, ProposedToolDraft
from app.schemas.tools import ToolDefinition

LOGGER = logging.getLogger("app.assistant.llm_client")

VIX_SYSTEM_PROMPT = """
You are Vix, a Windows desktop assistant.
Be concise, technical, and useful.
Do not claim you performed actions unless a tool result confirms it.
For real-time information, use tools instead of guessing.
For write or external actions, require permission.
""".strip()

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
        """Return a proposed tool spec for developer review only."""

    async def verify_connection(self) -> tuple[bool, str]:
        """Check whether the configured model endpoint is reachable."""


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

    async def verify_connection(self) -> tuple[bool, str]:
        return False, "OpenAI is not connected; using deterministic assistant fallbacks."


class OpenAILLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        client: object | None = None,
        tool_proposals_enabled: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client
        self._tool_proposals_enabled = tool_proposals_enabled

    async def complete(self, messages: list[dict[str, object]]) -> str:
        request_messages = _with_system_prompt(messages)
        LOGGER.info(
            "llm completion request | provider=openai | model=%s | messages=%s",
            self._model,
            len(request_messages),
        )
        response = await self._get_client().chat.completions.create(
            model=self._model,
            messages=request_messages,
        )
        content = response.choices[0].message.content
        if not content:
            LOGGER.warning("llm completion response was empty | provider=openai | model=%s", self._model)
            return "I could not generate an answer right now."
        LOGGER.info("llm completion raw output | provider=openai | model=%s | text=%s", self._model, _short_text(content, limit=4000))
        return str(content).strip()

    async def create_plan(self, user_text: str, tools: list[ToolDefinition]) -> AssistantPlan:
        return await DeterministicLLMClient().create_plan(user_text, tools)

    async def propose_tool_spec(self, user_message: str, existing_tools: list[ToolDefinition]) -> ProposedToolDraft | None:
        if not self._tool_proposals_enabled:
            return propose_missing_tool(user_message)

        LOGGER.info("ai proposal request | provider=openai | model=%s | message=%s", self._model, _short_text(user_message))
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
            LOGGER.warning("ai proposal response was empty | provider=openai | model=%s", self._model)
            return None
        LOGGER.info("ai proposal raw output | provider=openai | model=%s | json=%s", self._model, _short_text(content, limit=4000))
        draft = ProposedToolDraft.model_validate_json(content)
        LOGGER.info(
            "ai proposal validated | name=%s | risk=%s | description=%s | reason=%s",
            draft.name,
            draft.risk_level,
            _short_text(draft.description),
            _short_text(draft.reason),
        )
        return draft

    async def verify_connection(self) -> tuple[bool, str]:
        try:
            await self._get_client().models.retrieve(self._model)
        except Exception as exc:
            return False, f"OpenAI model verification failed: {redact_secrets(str(exc))}"
        return True, f"OpenAI model '{self._model}' is reachable."

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


def _with_system_prompt(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": VIX_SYSTEM_PROMPT}, *messages]


def _short_text(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def redact_secrets(value: str) -> str:
    value = re.sub(r"sk-[A-Za-z0-9_\\-]{8,}", "sk-***REDACTED***", value)
    return re.sub(r"(OPENAI_API_KEY=)[^\\s]+", r"\\1***REDACTED***", value)
