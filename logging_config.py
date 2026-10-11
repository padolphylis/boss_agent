from __future__ import annotations

import hashlib
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Generator


_context: ContextVar[dict[str, str]] = ContextVar(
    "boss_agent_log_context", default={}
)
_configure_lock = __import__("threading").Lock()
_configured = False


class _ContextFilter(logging.Filter):
    """为日志记录补齐 request_id/task_id 等可格式化字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _context.get()
        record.request_id = context.get("request_id", "-")
        record.task_id = context.get("task_id", "-")
        return True


def _env_int(name: str, default: int, minimum: int) -> int:
    """读取整型环境变量，缺失或非法时返回不小于 minimum 的默认值。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        return default


def fingerprint(value: Any, *, length: int = 8) -> str:
    """把任意值压成 ``len=<字符数> sha1=<前 N 位>`` 短摘要，仅用于日志留痕。"""
    if value is None:
        return "len=0 sha1=-"

    text = value if isinstance(value, str) else str(value)
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]
    return f"len={len(text)} sha1={digest}"


def _as_text(content: Any) -> str:
    """把 message.content 归一化成可读字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(parts)
    return str(content)


def summarize_message(message: Any) -> str:
    """描述一条模型响应：内容摘要 + 是否命中工具调用。"""
    content = _as_text(getattr(message, "content", ""))
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_names = ",".join(str(call.get("name", "?")) for call in tool_calls)
    return (
        f"content({fingerprint(content)}) "
        f"tool_calls={len(tool_calls)}[{tool_names}]"
    )


def configure_logging() -> None:
    """初始化全局日志配置；重复调用不会重复添加 handler。"""
    global _configured
    if _configured:
        return

    with _configure_lock:
        if _configured:
            return

        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        log_dir = Path(os.getenv("LOG_DIR", Path(__file__).parent / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)s | %(name)s | "
                "request_id=%(request_id)s task_id=%(task_id)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_filter = _ContextFilter()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(log_filter)

        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=_env_int("LOG_MAX_BYTES", 10 * 1024 * 1024, 1),
            backupCount=_env_int("LOG_BACKUP_COUNT", 5, 0),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(log_filter)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """返回已完成配置的模块 logger。"""
    configure_logging()
    return logging.getLogger(name)


def set_log_context(**values: str) -> Token:
    """设置当前上下文，并返回供 reset 使用的 token。"""
    current = _context.get()
    updated = {**current, **{key: str(value) for key, value in values.items()}}
    return _context.set(updated)


def reset_log_context(token: Token) -> None:
    """恢复 set_log_context 之前的上下文。"""
    _context.reset(token)


@contextmanager
def log_context(**values: str) -> Generator[None, None, None]:
    """在代码块内绑定 request_id/task_id 等日志上下文。"""
    token = set_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(token)
