import json
import sqlite3
from pathlib import Path

from app.db.models import EventRecord
from app.schemas.events import AssistantEvent


class Database:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self._path = Path(database_path) if database_path else Path(__file__).resolve().parents[2] / "assistant_events.sqlite3"
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def log_event(self, event: AssistantEvent) -> None:
        connection = self._ensure_connection()
        connection.execute(
            "INSERT INTO events (id, type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                event.id,
                event.type,
                json.dumps(event.payload, sort_keys=True),
                event.created_at.isoformat(),
            ),
        )
        connection.commit()

    def list_events(self, limit: int = 100) -> list[EventRecord]:
        connection = self._ensure_connection()
        rows = connection.execute(
            "SELECT id, type, payload_json, created_at FROM events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            EventRecord(
                id=row[0],
                type=row[1],
                payload_json=row[2],
                created_at=AssistantEvent.parse_created_at(row[3]),
            )
            for row in rows
        ]

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self.initialize()
        if self._connection is None:
            raise RuntimeError("Database connection was not initialized.")
        return self._connection
