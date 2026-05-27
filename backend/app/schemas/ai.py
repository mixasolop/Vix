from pydantic import BaseModel


class AIStatusResponse(BaseModel):
    provider: str
    model: str
    proposals_enabled: bool
    api_key_configured: bool
    connected: bool
    status: str
    detail: str
