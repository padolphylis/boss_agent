"""协作式暂停/继续控制，任务只会在明确的安全检查点暂停。"""

from threading import Condition
from time import monotonic
from typing import Callable


class TaskCancelled(Exception):
    """任务因客户端连接断开而取消。"""


class TaskControl:
    def __init__(self, on_change: Callable[[str], None] | None = None):
        self._condition = Condition()
        self._on_change = on_change
        self._pause_requested = False
        self._paused = False
        self._cancel_requested = False
        self._finished = False

    @property
    def status(self) -> str:
        with self._condition:
            if self._finished:
                return "finished"
            if self._cancel_requested:
                return "cancel_requested"
            if self._paused:
                return "paused"
            if self._pause_requested:
                return "pause_requested"
            return "running"

    def pause(self) -> str:
        with self._condition:
            if self._finished:
                return "finished"
            if self._pause_requested:
                return "paused" if self._paused else "pause_requested"
            self._pause_requested = True
            status = "pause_requested"
        self._notify(status)
        return status

    def resume(self) -> str:
        with self._condition:
            if self._finished:
                return "finished"
            if self._cancel_requested:
                return "cancel_requested"
            if not self._pause_requested:
                return "running"
            was_paused = self._paused
            self._pause_requested = False
            self._condition.notify_all()
            status = "resuming" if was_paused else "running"
        self._notify(status)
        return status

    def cancel(self) -> str:
        """请求任务停止；任务会在下一个安全检查点退出。"""
        with self._condition:
            if self._finished:
                return "finished"
            if self._cancel_requested:
                return "cancel_requested"
            self._cancel_requested = True
            self._pause_requested = False
            self._condition.notify_all()
        self._notify("cancel_requested")
        return "cancel_requested"

    def checkpoint(self) -> float:
        """阻塞至任务恢复；多个工作线程共享同一个暂停状态。"""
        with self._condition:
            if self._cancel_requested:
                raise TaskCancelled
            if self._finished or not self._pause_requested:
                return 0
            first_paused_worker = not self._paused
            self._paused = True
            paused_at = monotonic()
        if first_paused_worker:
            self._notify("paused")

        with self._condition:
            while self._pause_requested and not self._finished and not self._cancel_requested:
                self._condition.wait()
            if self._cancel_requested:
                self._paused = False
                raise TaskCancelled
            self._paused = False
            should_notify_running = not self._finished
        if should_notify_running:
            self._notify("running")
        return monotonic() - paused_at

    def finish(self) -> None:
        with self._condition:
            self._finished = True
            self._pause_requested = False
            self._cancel_requested = False
            self._condition.notify_all()

    def _notify(self, status: str) -> None:
        if self._on_change is not None:
            self._on_change(status)
