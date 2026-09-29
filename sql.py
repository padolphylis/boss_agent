"""SQLite 基础设施。

各业务模块分别维护自己的表结构和查询逻辑；本模块只负责 SQLite
连接的创建、线程安全事务和连接关闭，避免重复实现同一套生命周期代码。
"""

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator
import sqlite3


class SQLiteDatabase:
    """线程安全的轻量 SQLite 连接封装。"""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._closed = False

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """在锁内执行一组操作，并在退出时自动提交或回滚。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLite 数据库连接已关闭")
            with self._connection:
                yield self._connection

    def close(self) -> None:
        """关闭连接；重复关闭不会抛出异常。"""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True
