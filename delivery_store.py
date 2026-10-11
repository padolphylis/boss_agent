"""持久化职位投递状态，避免同一职位被重复投递。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sql import SQLiteDatabase


class DeliveryStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        lease_seconds: float = 30 * 60,
    ):
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "deliveries.db"
        self.lease_seconds = max(float(lease_seconds), 1.0)
        self._db = SQLiteDatabase(db_path)
        with self._db.transaction() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS job_deliveries (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    job_name TEXT NOT NULL DEFAULT '',
                    brand_name TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(job_deliveries)")
            }
            if "attempt_count" not in columns:
                db.execute(
                    "ALTER TABLE job_deliveries "
                    "ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "started_at" not in columns:
                db.execute("ALTER TABLE job_deliveries ADD COLUMN started_at TEXT")
            if "finished_at" not in columns:
                db.execute("ALTER TABLE job_deliveries ADD COLUMN finished_at TEXT")

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._db.transaction() as db:
            row = db.execute(
                "SELECT * FROM job_deliveries WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def reserve(self, job: dict, task_id: str = "") -> tuple[bool, str]:
        job_id = str(job.get("encryptJobId") or "").strip()
        if not job_id:
            return True, ""
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        with self._db.transaction() as db:
            row = db.execute(
                "SELECT status, attempt_count, started_at "
                "FROM job_deliveries WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                db.execute(
                    """
                    INSERT INTO job_deliveries(
                        job_id, status, message, job_name, brand_name, task_id,
                        updated_at, attempt_count, started_at, finished_at
                    ) VALUES (?, 'sending', '', ?, ?, ?, ?, 1, ?, NULL)
                    """,
                    (
                        job_id,
                        str(job.get("jobName") or ""),
                        str(job.get("brandName") or ""),
                        task_id,
                        now,
                        now,
                    ),
                )
                return True, ""

            status = str(row["status"] or "")
            if status == "success":
                return False, status
            if status in {"sending", "unknown"} and not self._is_stale(
                row["started_at"],
                now_dt,
            ):
                return False, status

            attempt_count = int(row["attempt_count"] or 0) + 1
            db.execute(
                """
                UPDATE job_deliveries
                SET status = 'sending',
                    message = '',
                    task_id = ?,
                    updated_at = ?,
                    attempt_count = ?,
                    started_at = ?,
                    finished_at = NULL
                WHERE job_id = ?
                """,
                (task_id, now, attempt_count, now, job_id),
            )
        return True, ""

    def update(self, job: dict, status: str, message: str = "", task_id: str = "") -> None:
        job_id = str(job.get("encryptJobId") or "").strip()
        if not job_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        finished_at = now if status != "sending" else None
        with self._db.transaction() as db:
            db.execute(
                """
                UPDATE job_deliveries
                SET status = ?, message = ?, task_id = ?, updated_at = ?,
                    finished_at = COALESCE(?, finished_at)
                WHERE job_id = ?
                """,
                (
                    status,
                    message,
                    task_id,
                    now,
                    finished_at,
                    job_id,
                ),
            )

    def _is_stale(self, started_at: str | None, now: datetime) -> bool:
        if not started_at:
            return True
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            return True
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (now - started).total_seconds() >= self.lease_seconds

    def close(self) -> None:
        self._db.close()
