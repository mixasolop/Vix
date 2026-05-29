from pydantic import BaseModel


class AIStatusResponse(BaseModel):
    provider: str
    model: str
    config_file_path: str
    general_answers_enabled: bool
    proposals_enabled: bool
    api_key_configured: bool
    api_key_fingerprint: str | None = None
    model_reachable: bool
    connected: bool
    status: str
    general_answers_status: str
    tool_proposals_status: str
    api_key_status: str
    model_status: str
    detail: str
    tool_execution_mode: str
