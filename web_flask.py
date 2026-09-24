import json
import atexit
import time
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4
from collections.abc import Iterator
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from body import run_task_stream
from browser import BROWSER_IO_LOCK, BrowserManager
from chat_worker import ChatWorker, auto_reply_enabled
from config import get as cfg, get_all, set_config
from conversation_store import ConversationStore
from logging_config import get_logger, log_context
from vector_store import vector_store

logger = get_logger(__name__)


browser = BrowserManager()
browser_lock = BROWSER_IO_LOCK
conversation_store = ConversationStore()
chat_worker = ChatWorker(browser, conversation_store, browser_lock)
pending_tasks: dict[str, tuple[str, str, str]] = {}
_task_events: dict[str, Queue] = {}
_task_threads: dict[str, Thread] = {}
_task_lock = Lock()


app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
UPLOAD_DIR = Path(__file__).parent / "data" / "resumes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CONFIG_FIELDS = {
    "chat_openai_api_key": "对话 API Key",
    "chat_openai_base_url": "对话 Base URL",
    "chat_openai_model": "对话模型",
    "embedding_openai_api_key": "Embedding API Key",
    "embedding_openai_base_url": "Embedding Base URL",
    "embedding_openai_model": "Embedding 模型",
    "auto_reply": "自动回复开关",
    "qdrant_enabled": "启用 Qdrant",
    "qdrant_url": "Qdrant 地址",
    "qdrant_api_key": "Qdrant API Key",
    "qdrant_path": "Qdrant 本地目录",
    "qdrant_collection": "Qdrant Collection",
}


def _qdrant_enabled() -> bool:
    """与向量库内部保持同一默认值：未配置时启用本地 Qdrant。"""
    return cfg("qdrant_enabled", "true").strip().lower() in {"1", "true", "yes", "on"}


def _public_config() -> dict[str, str | bool]:
    values = get_all()

    legacy_keys = {
        "chat_openai_api_key": "openai_api_key",
        "chat_openai_base_url": "openai_base_url",
        "chat_openai_model": "openai_model",
        "embedding_openai_api_key": "openai_api_key",
        "embedding_openai_base_url": "openai_base_url",
        "embedding_openai_model": "embedding_model",
    }

    def configured_value(key: str) -> str:
        legacy_key = legacy_keys.get(key)
        return values.get(key) or cfg(key) or (cfg(legacy_key) if legacy_key else "")

    config_values = {
        key: "已保存" if "api_key" in key and configured_value(key)
        else configured_value(key)
        for key in CONFIG_FIELDS
        if key not in {"auto_reply", "qdrant_enabled"}
    }
    return {
        "configured": bool(cfg("chat_openai_api_key") or cfg("openai_api_key"))
        and bool(cfg("embedding_openai_api_key") or cfg("openai_api_key")),
        "auto_reply": auto_reply_enabled(),
        "qdrant_enabled": _qdrant_enabled(),
        **config_values,
    }


def _resume_upload() -> str:
    """保存上传的 PDF/DOCX，返回服务端路径；无文件时返回空串。"""
    uploaded = request.files.get("resume_file")
    if uploaded is None or not uploaded.filename:
        return ""
    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ValueError("简历只支持 PDF 或 DOCX 文件。")
    if not original_name:
        raise ValueError("简历文件名无效。")
    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    uploaded.save(target)
    logger.info("简历上传成功: filename=%s size=%s", original_name, target.stat().st_size)
    return str(target)

def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _start_task(
    message: str,
    resume: str,
    task_id: str,
    conversation_id: str,
    snapshot: dict | None = None,
) -> Queue:
    events = Queue()
    conversation_store.save_task_snapshot(
        task_id,
        {
            "task_id": task_id,
            "user_input": message,
            "original_input": message,
            "resume_path": resume,
            "working": True,
            "status": "started",
        },
        conversation_id,
    )
    with _task_lock:
        _task_events[task_id] = events

    def worker() -> None:
        try:
            for node, state in run_task_stream(message, browser, resume, task_id, snapshot):
                if state is None:
                    continue
                conversation_store.save_task_snapshot(
                    task_id,
                    state.model_dump(mode="json"),
                    conversation_id,
                )
                events.put((node, state))
        except Exception as exc:
            logger.exception("后台任务失败: task_id=%s", task_id)
            events.put(("error", str(exc)))
        finally:
            events.put((None, None))
            with _task_lock:
                _task_threads.pop(task_id, None)

    thread = Thread(target=worker, name=f"task-{task_id[:8]}", daemon=True)
    with _task_lock:
        _task_threads[task_id] = thread
    thread.start()
    return events


def stream_chat(message: str, resume: str = "", task_id: str = "", conversation_id: str | None = None) -> Iterator[str]:
    """订阅后台任务，通过 SSE 返回节点状态和最终结果。"""
    request_id = str(uuid4())
    started = time.monotonic()
    with log_context(request_id=request_id):
        logger.info("收到聊天任务: request_id=%s resume_provided=%s", request_id, bool(resume))
        try:
            if not task_id:
                task_id = str(uuid4())
                events = _start_task(message, resume, task_id, conversation_id or "")
            else:
                with _task_lock:
                    events = _task_events.get(task_id)
                if events is None:
                    snapshot = conversation_store.task_snapshot(task_id)
                    if snapshot and snapshot["status"] not in {"completed", "need_input"}:
                        saved = snapshot["state"]
                        message = saved.get("original_input") or message
                        resume = saved.get("resume_path") or resume
                        events = _start_task(message, resume, task_id, conversation_id or snapshot["conversation_id"])
                    else:
                        raise ValueError("任务不存在或已经结束")

            yield _sse("status", {
                "stage": "start",
                "message": "正在初始化任务...",
                "task_id": task_id,
            })
            final_state = None
            while True:
                node, state = events.get()
                if node is None:
                    break
                if node == "error":
                    raise RuntimeError(state)
                stage_messages = {
                    "analyze_intent": "正在分析任务需求...",
                    "analyze_resume": "正在准备简历信息...",
                    "init_browser": "浏览器已准备就绪",
                    "check_login": "正在检查登录状态...",
                    "wait_login": "等待完成登录...",
                    "search_jobs": "正在读取职位页面...",
                    "match_job_content": "正在分析职位匹配度...",
                    "push_jobs": "正在执行投递...",
                }
                yield _sse("status", {
                    "stage": node,
                    "message": stage_messages.get(node, state.status or "处理中..."),
                })
                final_state = state

            if final_state is None:
                raise RuntimeError("任务未返回最终状态")
            logger.info(
                "聊天任务完成: request_id=%s task_id=%s status=%s duration=%.2fs",
                request_id,
                final_state.task_id,
                final_state.status,
                time.monotonic() - started,
            )
            if final_state.status == "need_input":
                pending_tasks[final_state.task_id] = (
                    final_state.original_input or message,
                    resume,
                    conversation_id or "",
                )
            elif task_id:
                pending_tasks.pop(task_id, None)
            if final_state.status == "need_input":
                result_message = final_state.pending_question or final_state.result or "请补充更多信息。"
            elif final_state.error:
                result_message = final_state.error
            else:
                result_message = final_state.result or "任务已完成。"
            conversation_store.save_web_message(
                conversation_id, str(uuid4()), "outgoing", result_message
            )
            yield _sse("result", {
                "message": result_message,
                "task_id": final_state.task_id,
                "status": final_state.status,
                "conversation_id": conversation_id,
                "search_params": final_state.search_params.model_dump(),
            })
            yield _sse("status", {"stage": "done", "message": "处理完成"})
        except Exception as exc:
            logger.exception("聊天任务失败: request_id=%s duration=%.2fs", request_id, time.monotonic() - started)
            yield _sse("error", {"message": f"任务执行失败：{type(exc).__name__}: {exc}"})



def _json_payload() -> dict:
    """读取 JSON 请求体，兼容空请求和错误 JSON。"""
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


@app.get("/")
def web():
    return render_template("web.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", chat_worker=chat_worker.status())


@app.get("/api/chat-worker")
def chat_worker_status():
    return jsonify(chat_worker.status())


@app.get("/api/tasks/<task_id>")
def task_status(task_id: str):
    snapshot = conversation_store.task_snapshot(task_id)
    if snapshot is None:
        return jsonify(error="任务不存在。"), 404
    return jsonify(snapshot)


@app.get("/api/tasks")
def task_list():
    return jsonify(tasks=conversation_store.resumable_tasks())


@app.get("/api/config")
def read_config():
    return jsonify(_public_config())


@app.get("/api/conversations")
def list_conversations():
    return jsonify(conversations=conversation_store.conversations())


@app.post("/api/conversations")
def create_conversation():
    conversation_id = str(uuid4())
    return jsonify(conversation_store.create_conversation(conversation_id)), 201


@app.get("/api/conversations/<conversation_id>/messages")
def conversation_messages(conversation_id: str):
    if not conversation_store.get_conversation(conversation_id):
        return jsonify(error="会话不存在。"), 404
    return jsonify(messages=conversation_store.history(conversation_id, limit=100))


@app.post("/api/config")
def write_config():
    payload = _json_payload()
    for key in CONFIG_FIELDS:
        value = payload.get(key)
        if "api_key" in key and (value is None or str(value).strip() in {"", "已保存"}):
            continue
        if value is not None:
            set_config(key, str(value).strip())
    logger.info(
        "收到配置更新请求: fields=%s",
        [key for key in CONFIG_FIELDS if key in payload and "api_key" not in key],
    )
    # 开关切换后立即生效：开启则拉起监听线程，关闭则停止监听。
    if auto_reply_enabled():
        chat_worker.start(True)
    else:
        chat_worker.stop()
    return jsonify(_public_config())


@app.post("/api/chat")
def chat():
    payload = request.form.to_dict() if request.form else _json_payload()
    message = str(payload.get("message", "")).strip()
    if not message:
        logger.warning("拒绝空聊天请求")
        return jsonify(error="请输入求职需求。"), 400

    conversation_id = str(payload.get("conversation_id", "")).strip()
    if not conversation_id:
        conversation_id = str(uuid4())
        conversation_store.create_conversation(conversation_id)
    elif not conversation_store.get_conversation(conversation_id):
        return jsonify(error="会话不存在。"), 404
    conversation_store.save_web_message(conversation_id, str(uuid4()), "incoming", message)

    resume = str(payload.get("resume", "")).strip()
    try:
        uploaded_resume = _resume_upload()
    except ValueError as exc:
        logger.warning("简历上传被拒绝: %s", exc)
        return jsonify(error=str(exc)), 400
    if uploaded_resume:
        resume = uploaded_resume

    task_id = str(payload.get("task_id", "")).strip()
    if task_id:
        original, saved_resume, saved_conversation_id = pending_tasks.get(task_id, ("", "", ""))
        snapshot = conversation_store.task_snapshot(task_id)
        if original:
            if saved_conversation_id != conversation_id:
                return jsonify(error="待补充任务不属于当前会话。"), 404
            message = f"{original}\n补充信息：{message}"
            resume = saved_resume
        elif not snapshot or snapshot["conversation_id"] != conversation_id:
            return jsonify(error="任务不存在或不属于当前会话。"), 404

    return Response(
        stream_with_context(stream_chat(message, resume, task_id, conversation_id)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


atexit.register(browser.close)
atexit.register(vector_store.close)
# 退出处理按注册逆序执行，因此监听器必须在浏览器关闭前停止。
atexit.register(chat_worker.stop)

# 本模块只负责定义 Flask 应用与共享实例，不提供直接运行入口。
# 启动服务请使用 python main.py：它会在启动时读取 auto_reply 配置并拉起聊天监听线程，
# 而直接运行本模块会跳过该步骤，导致自动回复静默失效。
