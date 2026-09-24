"""生成 Boss 对话回复。

默认只生成建议，不主动发送；真正发送由聊天桥接层的显式调用完成。
"""

import time

from langchain_openai import ChatOpenAI

from config import get as cfg
from logging_config import get_logger, fingerprint, summarize_message

logger = get_logger(__name__)


def generate_reply(history: list[dict]) -> str:
    options = {
        "model": cfg("chat_openai_model") or cfg("openai_model", "gpt-4o-mini"),
        "temperature": 0.2,
        "api_key": cfg("chat_openai_api_key") or cfg("openai_api_key"),
    }
    base_url = cfg("chat_openai_base_url") or cfg("openai_base_url")
    if base_url:
        options["base_url"] = base_url

    prompt = [
        {
            "role": "system",
            "content": (
                "你是求职者的沟通助手。回复要简短、礼貌、真实，不得虚构学历、经历、"
                "薪资、入职时间或其他个人信息；涉及薪资承诺、面试时间、隐私和附件时，"
                "只给出待确认建议，不要替用户做决定。"
            ),
        }
    ]
    for item in history:
        role = "assistant" if item.get("direction") == "outgoing" else "user"
        prompt.append({"role": role, "content": item.get("content", "")})

    # 对话历史属于隐私内容，只记条数与整体指纹，不落正文。
    logger.debug(
        "自动回复请求: model=%s messages=%s history(%s)",
        options["model"],
        len(prompt),
        fingerprint(str(history)),
    )

    started = time.monotonic()
    try:
        result = ChatOpenAI(**options).invoke(prompt)
    except Exception:
        logger.exception(
            "自动回复失败: model=%s messages=%s duration=%.2fs",
            options["model"],
            len(prompt),
            time.monotonic() - started,
        )
        raise

    reply = str(result.content).strip()
    logger.debug(
        "自动回复响应: model=%s response(%s) 空回复=%s duration=%.2fs",
        options["model"],
        summarize_message(result),
        not reply,
        time.monotonic() - started,
    )
    return reply
