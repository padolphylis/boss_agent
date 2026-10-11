from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import get as cfg
from logging_config import get_logger
from sql import SQLiteDatabase

logger = get_logger(__name__)

cache_schema_version = "v1"


def cache_enabled() -> bool:
    """读取缓存开关；默认启用，可用 ``semantic_cache_enabled=false`` 关闭。"""
    return cfg("semantic_cache_enabled", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


class SemanticDecisionCache:
    """基于 SQLite 的语义判断缓存，线程安全且按需连接。"""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            configured = cfg("semantic_cache_path").strip()
            db_path = configured or (
                Path(__file__).parent / "data" / "semantic_cache.db"
            )
        self._db_path = db_path
        self._db: SQLiteDatabase | None = None
        self._db_lock = Lock()
        self.enabled_override: bool | None = None

    @property
    def enabled(self) -> bool:
        if self.enabled_override is not None:
            return self.enabled_override
        return cache_enabled()

    def _get_db(self) -> SQLiteDatabase:
        with self._db_lock:
            if self._db is None:
                self._db = SQLiteDatabase(self._db_path)
                with self._db.transaction() as db:
                    db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS semantic_decisions (
                            cache_key TEXT PRIMARY KEY,
                            decision TEXT NOT NULL,
                            model TEXT NOT NULL DEFAULT '',
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                        """
                    )
            return self._db

    def get(self, cache_key: str) -> dict | None:
        """读取缓存；任何读取或解析异常都按未命中处理，不影响主流程。"""
        if not cache_key or not self.enabled:
            return None
        try:
            db = self._get_db()
            with db.transaction() as conn:
                row = conn.execute(
                    "SELECT decision FROM semantic_decisions "
                    "WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except Exception:
            logger.warning("读取语义缓存失败，按未命中处理", exc_info=True)
            return None
        if not row:
            return None
        try:
            value = json.loads(row[0])
        except Exception:
            logger.warning("语义缓存内容损坏，按未命中处理", exc_info=True)
            return None
        return value if isinstance(value, dict) else None

    def put(self, cache_key: str, decision: dict, model: str = "") -> None:
        """写入缓存；写入失败只记录日志，不中断语义审核。"""
        if not cache_key or not self.enabled:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            db = self._get_db()
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO semantic_decisions(
                        cache_key, decision, model, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        decision = excluded.decision,
                        model = excluded.model,
                        updated_at = excluded.updated_at
                    """,
                    (
                        cache_key,
                        json.dumps(decision, ensure_ascii=False),
                        model,
                        now,
                        now,
                    ),
                )
        except Exception:
            logger.warning("写入语义缓存失败，忽略", exc_info=True)

    def close(self) -> None:
        with self._db_lock:
            if self._db is not None:
                self._db.close()
                self._db = None
