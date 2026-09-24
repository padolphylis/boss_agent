"""聊天会话和消息的轻量持久化。

消息监听可能因为网络重连重复收到同一条消息，因此不能只依赖进程内集合去重。
这里使用 SQLite 保存会话游标和消息 ID，服务重启后仍能保持幂等。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class ConversationStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or Path(__file__).parent / "data" / "conversations.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    friend_id TEXT,
                    friend_name TEXT,
                    friend_source TEXT,
                    title TEXT NOT NULL DEFAULT '新对话',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sender_id TEXT,
                    sender_name TEXT,
                    direction TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    replied INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS task_snapshots (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(conversations)")}
            if "title" not in columns:
                self._db.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT '新对话'")

    def create_conversation(self, conversation_id: str, title: str = "新对话") -> dict[str, Any]:
        conversation_id = conversation_id.strip()
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO conversations(conversation_id, title) VALUES (?, ?)",
                (conversation_id, title.strip() or "新对话"),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT conversation_id, title, updated_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_web_message(self, conversation_id: str, message_id: str, direction: str, content: str) -> bool:
        saved = self.save_message({
            "message_id": message_id,
            "conversation_id": conversation_id,
            "direction": direction,
            "content": content,
            "raw": {"source": "web"},
        })
        if saved and direction == "incoming":
            with self._lock, self._db:
                self._db.execute(
                    "UPDATE conversations SET title = CASE WHEN title = '新对话' THEN ? ELSE title END, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                    (content[:40] or "新对话", conversation_id),
                )
        return saved

    def save_message(self, message: dict[str, Any]) -> bool:
        """保存消息；返回 False 表示 message_id 已存在。"""
        message_id = str(message.get("message_id") or "").strip()
        conversation_id = str(message.get("conversation_id") or "").strip()
        if not message_id or not conversation_id:
            return False

        with self._lock, self._db:
            cursor = self._db.execute(
                """
                INSERT OR IGNORE INTO messages
                (message_id, conversation_id, sender_id, sender_name, direction, content, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    str(message.get("sender_id") or ""),
                    str(message.get("sender_name") or ""),
                    str(message.get("direction") or "incoming"),
                    str(message.get("content") or ""),
                    json.dumps(message.get("raw") or message, ensure_ascii=False, default=str),
                ),
            )
            self._db.execute(
                """
                INSERT INTO conversations(conversation_id, friend_id, friend_name, friend_source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    friend_id=excluded.friend_id,
                    friend_name=excluded.friend_name,
                    friend_source=excluded.friend_source,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    conversation_id,
                    str(message.get("sender_id") or ""),
                    str(message.get("sender_name") or ""),
                    str(message.get("friend_source") or "0"),
                ),
            )
            return cursor.rowcount == 1

    def history(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._lock:
            rows = self._db.execute(
                """
                SELECT message_id, sender_id, sender_name, direction, content, created_at
                FROM messages WHERE conversation_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._db.execute(
                """
                SELECT conversation_id, friend_id, friend_name, friend_source, title, updated_at
                FROM conversations ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_task_snapshot(self, task_id: str, state: dict[str, Any], conversation_id: str = "") -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO task_snapshots(task_id, conversation_id, state_json, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    state_json=excluded.state_json,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    task_id,
                    conversation_id,
                    json.dumps(state, ensure_ascii=False, default=str),
                    str(state.get("status") or "running"),
                ),
            )

    def task_snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT task_id, conversation_id, state_json, status, updated_at "
                "FROM task_snapshots WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def resumable_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT task_id, conversation_id, state_json, status, updated_at "
                "FROM task_snapshots WHERE status NOT IN ('completed', 'need_input') "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["state"] = json.loads(item.pop("state_json"))
            result.append(item)
        return result

    def close(self) -> None:
        with self._lock:
            self._db.close()
