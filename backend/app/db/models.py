from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EventRecord:
    id: str
    session_id: str | None
    type: str
    data_json: str
    created_at: datetime


@dataclass(frozen=True)
class MessageRecord:
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    session_id: str | None
    run_id: str | None
    type: str
    title: str
    content_text: str | None
    data_json: str
    created_at: datetime


@dataclass(frozen=True)
class ClarificationRecord:
    id: str
    session_id: str
    run_id: str
    kind: str
    question: str
    proposed_tool_name: str | None
    proposed_arguments_json: str
    status: str
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None
