from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Any

from browser import BrowserManager
from boss_chat_auto_reply import generate_reply
from conversation_store import ConversationStore, boss_chat_conversation
from logging_config import get_logger

logger = get_logger(__name__)


def auto_reply_enabled() -> bool:
    """读取自动回复配置，仅显式的 true/1/yes/on 视为开启。"""
    from config import get

    return get("auto_reply").strip().lower() in {"true", "1", "yes", "on"}


class ChatWorker:
    def __init__(
        self,
        browser: BrowserManager,
        store: ConversationStore | None = None,
        browser_lock: Lock | None = None,
    ):
        self.browser = browser
        self.store = store or ConversationStore()
        self.browser_lock = browser_lock
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._listening = Event()
        self.auto_reply = False
        self.last_error = ""
        self.received_count = 0
        self.replied_count = 0

    @property
    def running(self) -> bool:
        """线程是否存活（包含已启动但仍在等待浏览器就绪的待命状态）。"""
        return bool(self._thread and self._thread.is_alive())

    @property
    def listening(self) -> bool:
        """是否已真正接入 Boss 聊天监听。"""
        return self._listening.is_set()

    def start(self, auto_reply: bool | None = None) -> None:
        """启动监听线程；未显式传入时以配置项 auto_reply 为准。"""
        with self._lock:
            if self.running:
                return
            self.auto_reply = auto_reply_enabled() if auto_reply is None else bool(auto_reply)
            self.last_error = ""
            self._stop.clear()
            self._thread = Thread(target=self._run, name="boss-chat-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.auto_reply = False
        self._stop_browser_listener()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3)
        if thread and thread.is_alive():
            logger.warning("聊天监听器停止超时")

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "listening": self.listening,
            "auto_reply": self.auto_reply,
            "received_count": self.received_count,
            "replied_count": self.replied_count,
            "last_error": self.last_error,
        }

    def _run(self) -> None:
        retry_delay = 1.0
        try:
            if not self._wait_for_browser():
                logger.info("Boss 聊天监听器在浏览器就绪前被停止")
                return

            while not self._stop.is_set():
                try:
                    self._with_browser_lock(self.browser.start)
                    self._with_browser_lock(self.browser.start_chat_listener)
                    self._listening.set()
                    retry_delay = 1.0
                    logger.info("Boss 聊天监听器已启动")
                    self._poll_messages_until_error()
                except Exception as exc:
                    self._listening.clear()
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    if self._stop.is_set():
                        break
                    logger.warning(
                        "Boss 聊天监听暂时中断，将在 %.1f 秒后重试: %s",
                        retry_delay,
                        self.last_error,
                    )
                    self._stop_browser_listener()
                    self._wait_or_stop(retry_delay)
                    retry_delay = min(retry_delay * 2, 16.0)
        finally:
            self._listening.clear()
            self._stop_browser_listener()
            logger.info("Boss 聊天监听器已停止")

    def _poll_messages_until_error(self) -> None:
        """持续消费消息；页面断开或 SDK 重置时让外层统一重连。"""
        while not self._stop.is_set():
            messages = self._with_browser_lock(
                lambda: self.browser.poll_chat_messages(timeout=1)
            )
            for message in messages:
                self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        """保存一条新消息，并将单条自动回复故障隔离在监听线程之外。"""
        if message.get("direction") != "incoming":
            return
        if not self.store.save_message(
            message,
            conversation_type=boss_chat_conversation,
        ):
            return
        self.received_count += 1
        if not self.auto_reply or not message.get("content"):
            return

        try:
            history = self.store.history(message["conversation_id"], limit=20)
            reply = generate_reply(history)
            if not reply.strip():
                logger.warning(
                    "自动回复为空，跳过发送: message_id=%s",
                    message.get("message_id"),
                )
                return
            self._with_browser_lock(
                lambda: self.browser.send_chat_message(
                    message["conversation"],
                    reply,
                )
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "自动回复处理失败，监听器继续运行: message_id=%s",
                message.get("message_id"),
            )
            return

        self.store.save_message(
            {
                "message_id": f"local-{message['message_id']}-reply",
                "conversation_id": message["conversation_id"],
                "sender_id": "self",
                "direction": "outgoing",
                "content": reply,
                "friend_source": message.get("friend_source", "0"),
                "conversation": message["conversation"],
                "raw": {"generated": True},
            },
            conversation_type=boss_chat_conversation,
        )
        self.replied_count += 1

    def _wait_or_stop(self, seconds: float, interval: float = 0.2) -> bool:
        """可中断退避等待，返回是否收到停止信号。"""
        deadline = seconds
        while deadline > 0 and not self._stop.is_set():
            delay = min(interval, deadline)
            self._stop.wait(delay)
            deadline -= delay
        return self._stop.is_set()

    def _wait_for_browser(self, poll_interval: float = 1.0) -> bool:
        """等待浏览器就绪，就绪返回 True，被停止返回 False。"""
        waited = False
        while not self._stop.is_set():
            if self.browser.is_ready() or self.browser.wait_until_ready(poll_interval):
                return True
            try:
                self._with_browser_lock(self.browser.start)
                if self.browser.is_ready():
                    return True
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.debug(
                    "等待浏览器时连接失败，下一轮重试: error=%s",
                    self.last_error,
                )
            if self._stop.wait(poll_interval):
                return False
            if not waited:
                logger.info("等待浏览器就绪后再启动 Boss 聊天监听")
                waited = True
        return False

    def _with_browser_lock(self, callback):
        if self.browser_lock is None:
            return callback()
        with self.browser_lock:
            return callback()

    def _stop_browser_listener(self) -> None:
        try:
            self._with_browser_lock(self.browser.stop_chat_listener)
        except Exception:
            logger.debug("停止聊天监听桥接失败", exc_info=True)
