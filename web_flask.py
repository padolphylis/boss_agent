import json
import atexit
from collections.abc import Iterator
from pathlib import Path
from threading import Lock

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from body import run_task
from browser import BrowserManager
from config import get as cfg, get_all, set_config


app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

CONFIG_FIELDS = {
    "openai_api_key": "OpenAI API Key",
    "openai_base_url": "OpenAI Base URL",
    "openai_model": "对话模型",
    "embedding_model": "向量模型",
}


def _public_config() -> dict[str, str | bool]:
    values = get_all()
    return {
        "configured": bool(cfg("openai_api_key")),
        "openai_api_key": "" if not values.get("openai_api_key") else "已保存",
        **{key: values.get(key, cfg(key)) for key in CONFIG_FIELDS if key != "openai_api_key"},
    }

browser = BrowserManager()
browser_lock = Lock()


def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_chat(message: str, resume: str = "") -> Iterator[str]:
    """执行 Agent 任务，通过 SSE 返回状态和最终结果。"""
    yield _sse("status", {"stage": "start", "message": "正在处理..."})
    try:
        with browser_lock:
            state = run_task(message, browser, resume)
        yield _sse("result", state.model_dump())
        yield _sse("status", {"stage": "done", "message": "处理完成"})
    except Exception as exc:
        yield _sse("error", {
            "message": f"任务执行失败：{type(exc).__name__}: {exc}",
        })


def _json_payload() -> dict:
    """读取 JSON 请求体，兼容空请求和错误 JSON。"""
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


@app.get("/")
def web():
    return render_template("web.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/config")
def read_config():
    return jsonify(_public_config())


@app.post("/api/config")
def write_config():
    payload = _json_payload()
    for key in CONFIG_FIELDS:
        value = payload.get(key)
        if key == "openai_api_key" and (value is None or str(value).strip() in {"", "已保存"}):
            continue
        if value is not None:
            set_config(key, str(value).strip())
    return jsonify(_public_config())


@app.post("/api/chat")
def chat():
    payload = _json_payload()
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify(error="请输入求职需求。"), 400

    resume = str(payload.get("resume", "")).strip()

    return Response(
        stream_with_context(stream_chat(message, resume)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


atexit.register(browser.close)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
