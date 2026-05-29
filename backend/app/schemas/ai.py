from pydantic import BaseModel


class AIStatusResponse(BaseModel):
    provider: str
    model: str
    config_file_path: str
    general_answers_enabled: bool
    proposals_enabled: bool
    api_key_configured: bool
    connected: bool
    status: str
    detail: str
    tool_execution_mode: str
