import uvicorn
from uvicorn.middleware.wsgi import WSGIMiddleware

from logging_config import configure_logging, get_logger
from web_flask import app, chat_worker
from chat_worker import auto_reply_enabled

logger = get_logger(__name__)


def _start_chat_worker() -> None:
    """仅在配置开启自动回复时启动监听线程，避免无意义占用浏览器。"""
    if auto_reply_enabled():
        chat_worker.start(True)
    else:
        logger.info("auto_reply 未开启，跳过聊天监听线程")


if __name__ == "__main__":
    configure_logging()
    _start_chat_worker()
    # 从 boss_agent 目录执行：python main.py
    uvicorn.run(
        WSGIMiddleware(app),
        host="127.0.0.1",
        port=5000,
        reload=False,
        log_config=None,
    )
