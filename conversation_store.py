from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sql import SQLiteDatabase

web_conversation = "web"
boss_chat_conversation = "boss_chat"
_conversation_types = frozenset({
    web_conversation,
    boss_chat_conversation,
})


class ConversationStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or Path(__file__).parent / "data" / "conversations.db")
        self._db = SQLiteDatabase(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._db.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    friend_id TEXT,
                    friend_name TEXT,
                    friend_source TEXT,
                    title TEXT NOT NULL DEFAULT '新对话',
                    preferences_json TEXT NOT NULL DEFAULT '{}',
                    conversation_type TEXT NOT NULL DEFAULT 'web',
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
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'running',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(conversations)")}
            if "title" not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT '新对话'")
            if "preferences_json" not in columns:
                db.execute(
                    "ALTER TABLE conversations "
                    "ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "conversation_type" not in columns:
                db.execute(
                    "ALTER TABLE conversations "
                    "ADD COLUMN conversation_type TEXT NOT NULL DEFAULT 'web'"
                )
            snapshot_columns = {
                row[1] for row in db.execute("PRAGMA table_info(task_snapshots)")
            }
            if "summary_json" not in snapshot_columns:
                db.execute(
                    "ALTER TABLE task_snapshots "
                    "ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'"
                )
                rows = db.execute(
                    "SELECT task_id, state_json FROM task_snapshots"
                ).fetchall()
                for row in rows:
                    try:
                        state = json.loads(row["state_json"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        state = {}
                    db.execute(
                        "UPDATE task_snapshots SET summary_json = ? WHERE task_id = ?",
                        (
                            json.dumps(
                                self._build_task_summary_state(state),
                                ensure_ascii=False,
                                default=str,
                            ),
                            row["task_id"],
                        ),
                    )
            self._migrate_conversation_types(db)

    @staticmethod
    def _normalize_conversation_type(value: str | None) -> str:
        conversation_type = str(value or web_conversation).strip()
        if conversation_type not in _conversation_types:
            raise ValueError(f"不支持的会话类型: {conversation_type}")
        return conversation_type

    @staticmethod
    def _migrate_conversation_types(db) -> None:
        """为旧版本记录补上网页/BOSS 会话边界。"""
        rows = db.execute(
            "SELECT conversation_id, raw_json FROM messages"
        ).fetchall()
        web_ids: set[str] = set()
        non_web_ids: set[str] = set()
        for row in rows:
            conversation_id = str(row["conversation_id"] or "").strip()
            if not conversation_id:
                continue
            try:
                raw = json.loads(row["raw_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                raw = {}
            if isinstance(raw, dict) and raw.get("source") == "web":
                web_ids.add(conversation_id)
            else:
                non_web_ids.add(conversation_id)

        boss_ids = non_web_ids - web_ids
        if not boss_ids:
            return
        db.executemany(
            """
            UPDATE conversations
            SET conversation_type = ?
            WHERE conversation_id = ? AND conversation_type = ?
            """,
            [
                (boss_chat_conversation, conversation_id, web_conversation)
                for conversation_id in boss_ids
            ],
        )

    def create_conversation(self, conversation_id: str, title: str = "新对话") -> dict[str, Any]:
        conversation_id = conversation_id.strip()
        with self._db.transaction() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO conversations(
                    conversation_id, title, conversation_type
                ) VALUES (?, ?, ?)
                """,
                (
                    conversation_id,
                    title.strip() or "新对话",
                    web_conversation,
                ),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(
        self,
        conversation_id: str,
        conversation_type: str = web_conversation,
    ) -> dict[str, Any] | None:
        conversation_type = self._normalize_conversation_type(conversation_type)
        with self._db.transaction() as db:
            row = db.execute(
                """
                SELECT conversation_id, title, conversation_type, updated_at
                FROM conversations
                WHERE conversation_id = ? AND conversation_type = ?
                """,
                (conversation_id, conversation_type),
            ).fetchone()
        return dict(row) if row else None

    def get_preferences(self, conversation_id: str) -> dict[str, Any]:
        """读取当前会话的结构化偏好；不存在或损坏时返回空对象。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return {}
        with self._db.transaction() as db:
            row = db.execute(
                """
                SELECT preferences_json FROM conversations
                WHERE conversation_id = ? AND conversation_type = ?
                """,
                (conversation_id, web_conversation),
            ).fetchone()
        if not row:
            return {}
        try:
            preferences = json.loads(row["preferences_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return preferences if isinstance(preferences, dict) else {}

    def save_preferences(
        self,
        conversation_id: str,
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        """覆盖保存一份会话偏好对象，不把偏好扩散到其他会话。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return {}
        normalized = dict(preferences or {})
        with self._db.transaction() as db:
            db.execute(
                """
                UPDATE conversations
                SET preferences_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ? AND conversation_type = ?
                """,
                (
                    json.dumps(normalized, ensure_ascii=False, default=str),
                    conversation_id,
                    web_conversation,
                ),
            )
        return normalized

    def save_web_message(self, conversation_id: str, message_id: str, direction: str, content: str) -> bool:
        saved = self.save_message(
            {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "direction": direction,
                "content": content,
                "raw": {"source": "web"},
            },
            conversation_type=web_conversation,
        )
        if saved and direction == "incoming":
            with self._db.transaction() as db:
                db.execute(
                    """
                    UPDATE conversations
                    SET title = CASE WHEN title = '新对话' THEN ? ELSE title END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE conversation_id = ? AND conversation_type = ?
                    """,
                    (content[:40] or "新对话", conversation_id, web_conversation),
                )
        return saved

    def save_message(
        self,
        message: dict[str, Any],
        *,
        conversation_type: str | None = None,
    ) -> bool:
        """保存消息；返回 False 表示 message_id 已存在。"""
        message_id = str(message.get("message_id") or "").strip()
        conversation_id = str(message.get("conversation_id") or "").strip()
        if not message_id or not conversation_id:
            return False
        if conversation_type is None:
            raw = message.get("raw")
            conversation_type = (
                web_conversation
                if isinstance(raw, dict) and raw.get("source") == "web"
                else boss_chat_conversation
            )
        conversation_type = self._normalize_conversation_type(conversation_type)

        with self._db.transaction() as db:
            cursor = db.execute(
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
            db.execute(
                """
                INSERT INTO conversations(
                    conversation_id, friend_id, friend_name, friend_source,
                    conversation_type
                )
                VALUES (?, ?, ?, ?, ?)
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
                    conversation_type,
                ),
            )
            return cursor.rowcount == 1

    def history(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._db.transaction() as db:
            rows = db.execute(
                """
                SELECT message_id, sender_id, sender_name, direction, content, created_at
                FROM messages WHERE conversation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._db.transaction() as db:
            rows = db.execute(
                """
                SELECT conversation_id, friend_id, friend_name, friend_source, title, updated_at
                FROM conversations
                WHERE conversation_type = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (web_conversation, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def conversation_task_snapshots(self, conversation_id: str) -> list[dict[str, Any]]:
        """返回会话关联的任务快照，删除会话时用于清理任务和临时文件。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return []
        with self._db.transaction() as db:
            rows = db.execute(
                "SELECT task_id, conversation_id, state_json, status, updated_at "
                "FROM task_snapshots WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["state"] = json.loads(item.pop("state_json"))
            result.append(item)
        return result

    def delete_conversation(
        self,
        conversation_id: str,
        conversation_type: str = web_conversation,
    ) -> bool:
        """删除指定类型会话及其消息、任务快照；默认只允许删除网页会话。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return False
        conversation_type = self._normalize_conversation_type(conversation_type)
        with self._db.transaction() as db:
            exists = db.execute(
                """
                SELECT 1 FROM conversations
                WHERE conversation_id = ? AND conversation_type = ?
                """,
                (conversation_id, conversation_type),
            ).fetchone()
            if not exists:
                return False
            db.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            db.execute(
                "DELETE FROM task_snapshots WHERE conversation_id = ?",
                (conversation_id,),
            )
            db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        return True

    def save_task_snapshot(self, task_id: str, state: dict[str, Any], conversation_id: str = "") -> None:
        summary = self._build_task_summary_state(state)
        with self._db.transaction() as db:
            db.execute(
                """
                INSERT INTO task_snapshots(
                    task_id, conversation_id, state_json, summary_json, status
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    state_json=excluded.state_json,
                    summary_json=excluded.summary_json,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    task_id,
                    conversation_id,
                    json.dumps(state, ensure_ascii=False, default=str),
                    json.dumps(summary, ensure_ascii=False, default=str),
                    str(state.get("status") or "running"),
                ),
            )

    @staticmethod
    def _build_task_summary_state(state: dict[str, Any]) -> dict[str, Any]:
        """在快照写入时生成小型状态视图，避免轮询反复解析完整结果。"""
        state = state if isinstance(state, dict) else {}
        progress = state.get("pipeline_progress") or {}
        push_results = state.get("push_results") or []
        state_fields = (
            "working",
            "status",
            "intent",
            "error",
            "error_source",
            "result",
            "pending_question",
            "original_input",
            "resume_filename",
            "resume_parse_status",
            "checkpoint_node",
            "pipeline_warning",
            "pipeline_progress",
            "conversation_preferences",
            "search_params",
        )
        return {
            "state": {
                key: state[key]
                for key in state_fields
                if key in state
            },
            "counts": {
                "jobs": progress.get("jobs", len(state.get("jobs") or [])),
                "details": progress.get(
                    "details",
                    len(state.get("job_cards") or []),
                ),
                "semantic_checked": progress.get(
                    "semantic_checked",
                    len(state.get("semantic_results") or []),
                ),
                "matched": progress.get(
                    "matched",
                    len(state.get("matched_jobs") or []),
                ),
                "delivery_success": progress.get(
                    "delivery_success",
                    sum(
                        1
                        for item in push_results
                        if item.get("success") and not item.get("skipped")
                    ),
                ),
            },
        }

    def task_snapshot_summary(self, task_id: str) -> dict[str, Any] | None:
        """仅读取任务摘要，不反序列化完整职位结果。"""
        with self._db.transaction() as db:
            row = db.execute(
                """
                SELECT task_id, conversation_id, status, updated_at, summary_json
                FROM task_snapshots WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            summary = json.loads(result.pop("summary_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            summary = {}
        if not isinstance(summary, dict):
            summary = {}
        result.update(summary)
        return result

    def task_snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self._db.transaction() as db:
            row = db.execute(
                "SELECT task_id, conversation_id, state_json, status, updated_at "
                "FROM task_snapshots WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def mark_running_tasks_interrupted(self) -> int:
        """服务启动时关闭无法确认仍在运行的旧任务。"""
        changed = 0
        with self._db.transaction() as db:
            rows = db.execute(
                """
                SELECT task_id, state_json
                FROM task_snapshots
                WHERE status NOT IN ('completed', 'cancelled', 'need_input', 'interrupted')
                """
            ).fetchall()
            for row in rows:
                try:
                    state = json.loads(row["state_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    state = {}
                if not isinstance(state, dict) or not state.get("working"):
                    continue
                state["working"] = False
                state["status"] = "interrupted"
                state["error"] = (
                    state.get("error")
                    or "服务重启时任务被中断，未自动恢复，避免重复投递。"
                )
                state["error_source"] = state.get("error_source") or "system"
                summary = self._build_task_summary_state(state)
                db.execute(
                    """
                    UPDATE task_snapshots
                    SET state_json = ?, summary_json = ?, status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = ?
                    """,
                    (
                        json.dumps(state, ensure_ascii=False, default=str),
                        json.dumps(summary, ensure_ascii=False, default=str),
                        "interrupted",
                        row["task_id"],
                    ),
                )
                changed += 1
        return changed

    def resumable_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._db.transaction() as db:
            rows = db.execute(
                "SELECT task_id, conversation_id, state_json, status, updated_at "
                "FROM task_snapshots "
                "WHERE status NOT IN ('completed', 'cancelled', 'interrupted') "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["state"] = json.loads(item.pop("state_json"))
            if (
                item["status"] != "need_input"
                and not item["state"].get("working")
            ):
                continue
            result.append(item)
            if len(result) >= max(1, min(int(limit), 100)):
                break
        return result

    def active_task(
        self,
        conversation_id: str,
        active_task_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """返回会话最近一条仍可恢复的任务快照。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return None
        with self._db.transaction() as db:
            rows = db.execute(
                """
                SELECT task_id, conversation_id, state_json, status, updated_at
                FROM task_snapshots
                WHERE conversation_id = ?
                  AND status NOT IN ('completed', 'cancelled')
                ORDER BY updated_at DESC
                """,
                (conversation_id,),
            ).fetchall()
        for row in rows:
            item = dict(row)
            item["state"] = json.loads(item.pop("state_json"))
            state = item["state"]
            if item["status"] == "need_input":
                return item
            if (
                state.get("working")
                and (
                    active_task_ids is None
                    or item["task_id"] in active_task_ids
                )
            ):
                return item
        return None

    def active_task_summary(
        self,
        conversation_id: str,
        active_task_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """读取会话最近可恢复任务的摘要，不载入职位详情和审计列表。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return None
        with self._db.transaction() as db:
            rows = db.execute(
                """
                SELECT task_id, conversation_id, status, updated_at, summary_json
                FROM task_snapshots
                WHERE conversation_id = ?
                  AND status NOT IN ('completed', 'cancelled', 'interrupted')
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                (conversation_id,),
            ).fetchall()
        for row in rows:
            item = dict(row)
            try:
                summary = json.loads(item.pop("summary_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                summary = {}
            if not isinstance(summary, dict):
                summary = {}
            state = summary.get("state") or {}
            if item["status"] == "need_input":
                item.update(summary)
                return item
            if (
                state.get("working")
                and (
                    active_task_ids is None
                    or item["task_id"] in active_task_ids
                )
            ):
                item.update(summary)
                return item
        return None

    def latest_task_summary(self, conversation_id: str) -> dict[str, Any] | None:
        """读取会话最近一次任务摘要，包括已经结束的任务。"""
        conversation_id = conversation_id.strip()
        if not conversation_id:
            return None
        with self._db.transaction() as db:
            row = db.execute(
                """
                SELECT task_id, conversation_id, status, updated_at, summary_json
                FROM task_snapshots
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            summary = json.loads(item.pop("summary_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            summary = {}
        if not isinstance(summary, dict):
            summary = {}
        item.update(summary)
        return item

    def close(self) -> None:
        self._db.close()
