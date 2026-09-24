"""持久化职位投递状态，避免同一职位被重复投递。"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class DeliveryStore:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "deliveries.db"
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._lock, self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS job_deliveries (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    job_name TEXT NOT NULL DEFAULT '',
                    brand_name TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM job_deliveries WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def reserve(self, job: dict, task_id: str = "") -> tuple[bool, str]:
        job_id = str(job.get("encryptJobId") or "").strip()
        if not job_id:
            return True, ""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._db:
            inserted = self._db.execute(
                """
                INSERT OR IGNORE INTO job_deliveries(
                    job_id, status, message, job_name, brand_name, task_id, updated_at
                ) VALUES (?, 'sending', '', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(job.get("jobName") or ""),
                    str(job.get("brandName") or ""),
                    task_id,
                    now,
                ),
            ).rowcount
            if inserted:
                return True, ""
            row = self._db.execute(
                "SELECT status FROM job_deliveries WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row and row[0] in {"success", "sending", "unknown"}:
                return False, row[0]
            self._db.execute(
                """
                UPDATE job_deliveries
                SET status = 'sending', message = '', task_id = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (task_id, now, job_id),
            )
        return True, ""

    def update(self, job: dict, status: str, message: str = "", task_id: str = "") -> None:
        job_id = str(job.get("encryptJobId") or "").strip()
        if not job_id:
            return
        with self._lock, self._db:
            self._db.execute(
                """
                UPDATE job_deliveries
                SET status = ?, message = ?, task_id = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    message,
                    task_id,
                    datetime.now(timezone.utc).isoformat(),
                    job_id,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()
