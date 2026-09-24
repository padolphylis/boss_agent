"""统一配置层。

优先从 SQLite 读取（前端可动态修改），查不到则回退到 .env 环境变量。
GitHub 用户零配置：不连数据库也能跑。

用法：
    from config import get
    api_key = get("openai_api_key")

    # 前端改配置
    from config import set_config
    set_config("openai_model", "gpt-4o")
"""

import os
import logging
import sqlite3
from pathlib import Path

from logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_db: sqlite3.Connection | None = None


def init_db(db_path: str | Path | None = None) -> None:
    """启用 SQLite 配置存储。不调用则只读 .env。"""
    global _db
    if db_path is None:
        db_path = Path(__file__).parent / "data" / "config.db"
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(str(db_path), check_same_thread=False)
    _db.execute(
        "CREATE TABLE IF NOT EXISTS settings "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    _db.commit()
    logger.info("配置数据库已初始化: path=%s", db_path)


def get(key: str, default: str = "") -> str:
    """读配置：先数据库，后 .env。"""
    if _db:
        row = _db.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return row[0]
    return os.getenv(key, default)


def set_config(key: str, value: str) -> None:
    """写配置到数据库（前端接口调用）。"""
    if _db is None:
        init_db()
    assert _db is not None
    _db.execute(
        "REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
    )
    _db.commit()
    logger.info("配置已更新: key=%s", key)


def get_all() -> dict[str, str]:
    """返回数据库里所有配置（不含 .env）。"""
    if not _db:
        return {}
    rows = _db.execute("SELECT key, value FROM settings").fetchall()
    return {k: v for k, v in rows}


# 模块导入时自动加载 .env，确保环境变量就绪。
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.debug("未安装 python-dotenv，继续使用系统环境变量")


_load_env()
init_db()
