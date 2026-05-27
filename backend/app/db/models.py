from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EventRecord:
    id: str
    type: str
    payload_json: str
    created_at: datetime


@dataclass(frozen=True)
class MessageRecord:
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
