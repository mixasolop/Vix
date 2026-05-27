import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.db.models import EventRecord, MessageRecord
from app.schemas.events import AssistantEvent


class Database:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self._path = Path(database_path) if database_path else Path(__file__).resolve().parents[2] / "assistant_events.sqlite3"
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        self._ensure_column("events", "session_id", "TEXT")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS permissions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
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
        session_id = self._session_id_from_payload(event.payload)
        if session_id is not None:
            self.ensure_session(session_id)
        connection.execute(
            "INSERT INTO events (id, session_id, type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                event.id,
                session_id,
                event.type,
                json.dumps(event.payload, sort_keys=True),
                event.created_at.isoformat(),
            ),
        )
        connection.commit()

    def ensure_session(self, session_id: str, title: str | None = None) -> None:
        connection = self._ensure_connection()
        now = self._now_iso()
        connection.execute(
            """
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = COALESCE(excluded.title, sessions.title),
                updated_at = excluded.updated_at
            """,
            (session_id, title, now, now),
        )
        connection.commit()

    def log_message(self, session_id: str, role: str, content: str) -> str:
        self.ensure_session(session_id)
        message_id = str(uuid4())
        connection = self._ensure_connection()
        connection.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, self._now_iso()),
        )
        connection.commit()
        return message_id

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, object],
        status: str,
        result: dict[str, object],
        error: str | None = None,
    ) -> str:
        self.ensure_session(session_id)
        tool_call_id = str(uuid4())
        connection = self._ensure_connection()
        connection.execute(
            """
            INSERT INTO tool_calls
                (id, session_id, tool_name, arguments_json, status, result_json, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tool_call_id,
                session_id,
                tool_name,
                json.dumps(arguments, sort_keys=True),
                status,
                json.dumps(result, sort_keys=True),
                error,
                self._now_iso(),
            ),
        )
        connection.commit()
        return tool_call_id

    def create_permission(
        self,
        permission_id: str,
        session_id: str | None,
        tool_name: str,
        reason: str,
        arguments: dict[str, object],
        status: str = "pending",
    ) -> None:
        if session_id is not None:
            self.ensure_session(session_id)
        connection = self._ensure_connection()
        connection.execute(
            """
            INSERT INTO permissions
                (id, session_id, tool_name, status, reason, arguments_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                reason = excluded.reason,
                arguments_json = excluded.arguments_json
            """,
            (
                permission_id,
                session_id,
                tool_name,
                status,
                reason,
                json.dumps(arguments, sort_keys=True),
                self._now_iso(),
            ),
        )
        connection.commit()

    def update_permission_status(self, permission_id: str, status: str) -> None:
        connection = self._ensure_connection()
        connection.execute(
            "UPDATE permissions SET status = ?, decided_at = ? WHERE id = ?",
            (status, self._now_iso(), permission_id),
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

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        connection = self._ensure_connection()
        rows = connection.execute(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [
            MessageRecord(
                id=row[0],
                session_id=row[1],
                role=row[2],
                content=row[3],
                created_at=AssistantEvent.parse_created_at(row[4]),
            )
            for row in rows
        ]

    def count_rows(self, table_name: str) -> int:
        if table_name not in {"sessions", "messages", "events", "tool_calls", "permissions"}:
            raise ValueError(f"Unsupported table name: {table_name}")
        connection = self._ensure_connection()
        return int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self.initialize()
        if self._connection is None:
            raise RuntimeError("Database connection was not initialized.")
        return self._connection

    def _ensure_column(self, table_name: str, column_name: str, column_definition: str) -> None:
        connection = self._ensure_connection()
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _session_id_from_payload(payload: dict[str, object]) -> str | None:
        value = payload.get("conversation_id") or payload.get("session_id")
        return value if isinstance(value, str) and value else None
