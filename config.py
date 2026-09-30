
import os
import logging
from pathlib import Path

from logging_config import configure_logging, get_logger
from sql import SQLiteDatabase

configure_logging()
logger = get_logger(__name__)

_db: SQLiteDatabase | None = None


def init_db(db_path: str | Path | None = None) -> None:
    """启用 SQLite 配置存储。不调用则只读 .env。"""
    global _db
    if db_path is None:
        db_path = Path(__file__).parent / "data" / "config.db"
    if _db is not None:
        _db.close()
    _db = SQLiteDatabase(db_path)
    with _db.transaction() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS settings "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    logger.info("配置数据库已初始化: path=%s", db_path)


def get(key: str, default: str = "") -> str:
    """读配置：先数据库，后 .env。"""
    database = _db
    if database is not None:
        with database.transaction() as db:
            row = db.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row:
            return row[0]
    return os.getenv(key, default)


def set_config(key: str, value: str) -> None:
    """写配置到数据库（前端接口调用）。"""
    if _db is None:
        init_db()
    database = _db
    assert database is not None
    with database.transaction() as db:
        db.execute(
            "REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
        )
    logger.info("配置已更新: key=%s", key)


def get_all() -> dict[str, str]:
    """返回数据库里所有配置（不含 .env）。"""
    database = _db
    if database is None:
        return {}
    with database.transaction() as db:
        rows = db.execute("SELECT key, value FROM settings").fetchall()
    return {k: v for k, v in rows}


def close_db() -> None:
    """关闭配置数据库连接。"""
    global _db
    if _db is not None:
        _db.close()
        _db = None


# 模块导入时自动加载 .env，确保环境变量就绪。
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.debug("未安装 python-dotenv，继续使用系统环境变量")


_load_env()
init_db()
