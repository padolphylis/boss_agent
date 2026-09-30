import json
import atexit
import time
from queue import Empty, Queue
from threading import Lock, Thread
from uuid import uuid4
from collections.abc import Iterator
from pathlib import Path
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from body import run_task_stream
from browser import BROWSER_IO_LOCK, BrowserManager
from chat_worker import ChatWorker, auto_reply_enabled
from config import close_db, get as cfg, get_all, set_config
from conversation_store import ConversationStore
from logging_config import get_logger, log_context
from task_control import TaskControl
from vector_store import vector_store

logger = get_logger(__name__)


browser = BrowserManager()
browser_lock = BROWSER_IO_LOCK
conversation_store = ConversationStore()
chat_worker = ChatWorker(browser, conversation_store, browser_lock)
pending_tasks: dict[str, tuple[str, str, str, str]] = {}
_task_events: dict[str, Queue] = {}
_task_threads: dict[str, Thread] = {}
_task_controls: dict[str, TaskControl] = {}
_task_lock = Lock()


app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
UPLOAD_DIR = Path(__file__).parent / "data" / "resumes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_RESUME_SIZE_BYTES = 30 * 1024 * 1024
# multipart/form-data 还会包含字段名和边界，给请求头预留少量空间。
app.config["MAX_CONTENT_LENGTH"] = MAX_RESUME_SIZE_BYTES + 512 * 1024


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_exc):
    return jsonify(
        error="上传请求超过大小限制。简历文件上限为 30MB，请压缩文件后重试。"
    ), 413

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


def _resume_upload() -> tuple[str, str]:
    """保存上传的简历，返回服务端路径和用户可见文件名。"""
    uploaded = request.files.get("resume_file")
    if uploaded is None or not uploaded.filename:
        return "", ""
    display_name = uploaded.filename.replace("\\", "/").rsplit("/", 1)[-1]
    safe_name = secure_filename(display_name)
    suffix = Path(display_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc"}:
        raise ValueError("简历只支持 PDF、DOCX 或 DOC 文件。")
    if not safe_name:
        raise ValueError("简历文件名无效。")
    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    try:
        uploaded.save(target)
        size = target.stat().st_size
        if size > MAX_RESUME_SIZE_BYTES:
            raise RequestEntityTooLarge(
                description=(
                    f"简历文件过大（{size / 1024 / 1024:.1f}MB），"
                    "请将文件控制在 30MB 以内。"
                )
            )
    except Exception:
        target.unlink(missing_ok=True)
        raise
    logger.info("简历上传成功: filename=%s size=%s", safe_name, size)
    return str(target), display_name


def _conversation_has_running_task(conversation_id: str) -> bool:
    """删除会话前检查是否有后台任务仍在运行，避免任务完成时把会话重新写回来。"""
    with _task_lock:
        running_task_ids = tuple(_task_threads)
    for task_id in running_task_ids:
        snapshot = conversation_store.task_snapshot(task_id)
        if snapshot and snapshot["conversation_id"] == conversation_id:
            return True
    return False


def _remove_conversation_resume_files(snapshots: list[dict]) -> None:
    """删除会话关联的上传简历，仅允许删除 data/resumes 下的文件。"""
    upload_root = UPLOAD_DIR.resolve()
    for snapshot in snapshots:
        resume_path = str(snapshot.get("state", {}).get("resume_path") or "").strip()
        if not resume_path:
            continue
        candidate = Path(resume_path)
        try:
            candidate.resolve().relative_to(upload_root)
        except ValueError:
            logger.warning("跳过会话外的简历文件清理: path=%s", resume_path)
            continue
        candidate.unlink(missing_ok=True)


def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resume_status(state, node: str = "") -> str:
    """把内部任务状态转换成前端可直接展示的简历状态。"""
    if not state.resume_filename and not state.resume_path:
        return "none"
    if state.resume_parse_status in {"parsed", "error", "skipped"}:
        return state.resume_parse_status
    if node == "analyze_resume" or (
        node == "analyze_intent" and state.status == "intent_analyzed"
    ):
        return "parsing"
    return "uploaded"


def _task_status_payload(node: str, state) -> dict:
    """把节点状态转换成前端可以直接展示和处理的反馈。"""
    status = state.status or ""
    stage_messages = {
        "analyze_intent": "正在分析任务需求...",
        "analyze_resume": "正在解析简历...",
        "prepare_recommendation": "正在整理推荐条件...",
        "analyze_content": "正在生成分析报告...",
        "init_browser": "浏览器已准备就绪",
        "check_login": "正在检查登录状态...",
        "wait_login": "正在等待登录完成...",
        "search_jobs": "正在读取职位页面...",
        "match_job_content": "正在分析职位匹配度...",
        "recommendation_result": "正在整理推荐结果...",
        "push_jobs": "正在执行投递...",
    }
    message = stage_messages.get(node, status or "处理中...")
    payload = {
        "stage": node,
        "message": message,
        "status": status,
        "severity": "info",
        "resume_filename": state.resume_filename,
        "resume_status": _resume_status(state, node),
    }
    if node == "analyze_intent" and state.intent:
        payload["intent"] = state.intent
        payload["search_params"] = state.search_params.model_dump()

    if node == "check_login" and status == "login_required":
        # check_login 完成后，wait_login 节点会持续轮询很长时间；
        # 先把需要用户操作的信息推给前端，避免页面长时间没有变化。
        payload.update(
            stage="wait_login",
            message=(
                "需要登录：请在已打开的 Boss 直聘浏览器窗口完成登录。"
                "登录完成后系统会自动继续，无需重新发送。"
            ),
            severity="action",
            action_required=True,
            action="login",
        )
    elif state.error:
        payload.update(
            message=state.error,
            severity="error",
        )
    elif node == "init_browser" and status == "browser_ready":
        payload.update(
            stage="check_login",
            message="正在检查登录状态...",
        )
    elif node == "check_login" and status == "login_verified":
        payload.update(
            message="登录状态已确认，正在读取职位页面...",
        )
    elif node == "wait_login" and status == "login_verified":
        payload.update(
            message="登录已完成，正在读取职位页面...",
        )
    elif node == "search_jobs" and status == "search_timeout":
        if state.intent == "job_recommendation":
            payload.update(
                stage="recommendation_result",
                message=(
                    state.pipeline_warning
                    or "职位匹配超时，正在整理已完成的推荐结果..."
                ),
            )
        else:
            payload.update(
                stage="push_jobs",
                message=(
                    state.pipeline_warning
                    or "职位匹配超时，正在使用已完成的匹配结果执行投递..."
                ),
            )
    elif node == "search_jobs":
        payload.update(
            stage="match_job_content",
            message="职位已读取，正在分析匹配度...",
        )
    elif node == "match_job_content":
        if state.intent == "job_recommendation":
            payload.update(
                stage="recommendation_result",
                message="匹配完成，正在整理推荐结果...",
            )
        else:
            payload.update(
                stage="push_jobs",
                message="匹配完成，正在执行投递...",
            )
    elif node == "recommendation_result":
        payload["message"] = "推荐结果已生成。"
    elif node == "push_jobs":
        payload["message"] = "投递处理完成。"

    return payload


def _start_task(
    message: str,
    resume: str,
    task_id: str,
    conversation_id: str,
    snapshot: dict | None = None,
    resume_filename: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> Queue:
    events = Queue()
    saved_history = (
        snapshot.get("conversation_history")
        if snapshot and snapshot.get("conversation_history") is not None
        else (conversation_history or [])
    )
    conversation_store.save_task_snapshot(
        task_id,
        {
            "task_id": task_id,
            "user_input": message,
            "original_input": message,
            "resume_path": resume,
            "resume_filename": resume_filename,
            "resume_parse_status": "uploaded" if resume else "none",
            "conversation_history": saved_history,
            "working": True,
            "status": "started",
        },
        conversation_id,
    )
    with _task_lock:
        _task_events[task_id] = events

    task_control = TaskControl(
        on_change=lambda status: events.put((
            "task_control",
            {"task_id": task_id, "status": status},
        ))
    )

    def worker() -> None:
        try:
            for node, state in run_task_stream(
                message,
                browser,
                resume,
                task_id,
                snapshot,
                resume_filename,
                saved_history,
                task_control=task_control,
            ):
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
            task_control.finish()
            events.put((None, None))
            with _task_lock:
                _task_threads.pop(task_id, None)
                _task_controls.pop(task_id, None)
                _task_events.pop(task_id, None)

    thread = Thread(target=worker, name=f"task-{task_id[:8]}", daemon=True)
    with _task_lock:
        _task_threads[task_id] = thread
        _task_controls[task_id] = task_control
    thread.start()
    return events


def stream_chat(
    message: str,
    resume: str = "",
    task_id: str = "",
    conversation_id: str | None = None,
    resume_filename: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    """订阅后台任务，通过 SSE 返回节点状态和最终结果。"""
    request_id = str(uuid4())
    started = time.monotonic()
    with log_context(request_id=request_id):
        logger.info("收到聊天任务: request_id=%s resume_provided=%s", request_id, bool(resume))
        try:
            if not task_id:
                task_id = str(uuid4())
                events = _start_task(
                    message,
                    resume,
                    task_id,
                    conversation_id or "",
                    resume_filename=resume_filename,
                    conversation_history=conversation_history,
                )
            else:
                with _task_lock:
                    events = _task_events.get(task_id)
                if events is None:
                    snapshot = conversation_store.task_snapshot(task_id)
                    if (
                        snapshot
                        and snapshot["status"] not in {"completed", "need_input"}
                        and snapshot["state"].get("working", True)
                    ):
                        saved = snapshot["state"]
                        message = saved.get("original_input") or message
                        resume = saved.get("resume_path") or resume
                        resume_filename = saved.get("resume_filename") or resume_filename
                        events = _start_task(
                            message,
                            resume,
                            task_id,
                            conversation_id or snapshot["conversation_id"],
                            snapshot=snapshot["state"],
                            resume_filename=resume_filename,
                            conversation_history=conversation_history,
                        )
                    else:
                        raise ValueError("任务不存在或已经结束")

            yield _sse("status", {
                "stage": "start",
                "message": "正在初始化任务...",
                "task_id": task_id,
                "resume_filename": resume_filename,
                "resume_status": "uploaded" if resume else "none",
                "task_control": "running",
            })
            final_state = None
            waiting_for_action = False
            while True:
                try:
                    node, state = events.get(timeout=15)
                except Empty:
                    if waiting_for_action:
                        yield _sse("status", {
                            "stage": "wait_login",
                            "message": (
                                "仍在等待登录完成。请在已打开的 Boss 直聘浏览器窗口登录，"
                                "登录后系统会自动继续。"
                            ),
                            "status": "login_required",
                            "severity": "action",
                            "action_required": True,
                            "action": "login",
                        })
                    else:
                        # 避免长任务经过代理或浏览器时因无输出而断开连接。
                        yield ": keep-alive\n\n"
                    continue
                if node is None:
                    break
                if node == "error":
                    raise RuntimeError(state)
                if node == "task_control":
                    yield _sse("task_control", state)
                    continue
                status_payload = _task_status_payload(node, state)
                if node == "analyze_resume":
                    status_payload["message"] = {
                        "resume_analyzed": "简历已解析",
                        "resume_error": "简历解析失败",
                        "resume_skipped": "未上传简历，跳过解析",
                    }.get(state.status, status_payload["message"])
                yield _sse("status", status_payload)
                waiting_for_action = bool(status_payload.get("action_required"))
                if status_payload.get("action_required"):
                    yield _sse("action_required", {
                        "action": status_payload["action"],
                        "message": status_payload["message"],
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
                    final_state.resume_filename,
                    conversation_id or "",
                )
            elif task_id:
                pending_tasks.pop(task_id, None)
            if final_state.error:
                result_message = final_state.error
            elif final_state.status == "need_input":
                result_message = final_state.pending_question or final_state.result or "请补充更多信息。"
            else:
                result_message = final_state.result or "任务已完成。"
            conversation_store.save_web_message(
                conversation_id, str(uuid4()), "outgoing", result_message
            )
            yield _sse("result", {
                "message": result_message,
                "task_id": final_state.task_id,
                "status": final_state.status,
                "severity": "error" if final_state.error else "success",
                "action_required": final_state.status == "need_input",
                "intent": final_state.intent,
                "conversation_id": conversation_id,
                "search_params": final_state.search_params.model_dump(),
                "resume_filename": final_state.resume_filename,
                "resume_status": _resume_status(final_state),
            })
            yield _sse("status", {
                "stage": "done",
                "message": "处理完成",
                "status": final_state.status,
                "severity": "error" if final_state.error else "success",
            })
        except (BrokenPipeError, ConnectionResetError, GeneratorExit):
            with _task_lock:
                control = _task_controls.get(task_id)
            if control is not None:
                status = control.cancel()
                logger.info(
                    "SSE 连接断开，已请求终止后台任务: task_id=%s status=%s",
                    task_id,
                    status,
                )
            raise
        except Exception as exc:
            logger.exception("聊天任务失败: request_id=%s duration=%.2fs", request_id, time.monotonic() - started)
            error_message = f"任务执行失败：{type(exc).__name__}: {exc}"
            try:
                conversation_store.save_web_message(
                    conversation_id,
                    str(uuid4()),
                    "outgoing",
                    error_message,
                )
            except Exception:
                logger.exception("保存任务失败消息失败: request_id=%s", request_id)
            yield _sse("error", {
                "message": error_message,
                "status": "failed",
                "severity": "error",
                "terminal": True,
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


def _set_task_control(task_id: str, action: str):
    with _task_lock:
        control = _task_controls.get(task_id)
    if control is None:
        return jsonify(error="任务不存在或已经结束。"), 404

    status = control.pause() if action == "pause" else control.resume()
    if status == "finished":
        return jsonify(error="任务已经结束。"), 409
    return jsonify(task_id=task_id, status=status)


@app.post("/api/tasks/<task_id>/pause")
def pause_task(task_id: str):
    return _set_task_control(task_id, "pause")


@app.post("/api/tasks/<task_id>/resume")
def resume_task(task_id: str):
    return _set_task_control(task_id, "resume")


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


@app.delete("/api/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    conversation_id = conversation_id.strip()
    if not conversation_store.get_conversation(conversation_id):
        return jsonify(error="会话不存在。"), 404
    if _conversation_has_running_task(conversation_id):
        return jsonify(error="当前会话仍有任务执行，请等待任务结束后再删除。"), 409

    snapshots = conversation_store.conversation_task_snapshots(conversation_id)
    with _task_lock:
        pending_task_ids = [
            task_id
            for task_id, pending in pending_tasks.items()
            if pending[3] == conversation_id
        ]
        for task_id in pending_task_ids:
            pending_tasks.pop(task_id, None)

    if not conversation_store.delete_conversation(conversation_id):
        return jsonify(error="会话不存在。"), 404
    _remove_conversation_resume_files(snapshots)
    logger.info("会话已删除: conversation_id=%s", conversation_id)
    return jsonify(deleted=True, conversation_id=conversation_id)


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
    # 当前消息随后才会写入数据库，先取历史可避免意图模型收到当前消息两次。
    conversation_history = conversation_store.history(conversation_id, limit=12)
    conversation_store.save_web_message(conversation_id, str(uuid4()), "incoming", message)

    resume = str(payload.get("resume", "")).strip()
    resume_filename = str(payload.get("resume_filename", "")).strip()
    try:
        uploaded_resume, uploaded_filename = _resume_upload()
    except ValueError as exc:
        logger.warning("简历上传被拒绝: %s", exc)
        return jsonify(error=str(exc)), 400
    if uploaded_resume:
        resume = uploaded_resume
        resume_filename = uploaded_filename

    task_id = str(payload.get("task_id", "")).strip()
    if task_id:
        original, saved_resume, saved_resume_filename, saved_conversation_id = pending_tasks.get(
            task_id,
            ("", "", "", ""),
        )
        snapshot = conversation_store.task_snapshot(task_id)
        if original:
            if saved_conversation_id != conversation_id:
                return jsonify(error="待补充任务不属于当前会话。"), 404
            message = f"{original}\n补充信息：{message}"
            resume = saved_resume
            resume_filename = saved_resume_filename
            # 澄清回答要启动新任务。旧任务的事件队列已经消费完，
            # 继续复用旧 task_id 会拿到空队列，无法得到最终状态。
            pending_tasks.pop(task_id, None)
            task_id = ""
        elif snapshot and snapshot["conversation_id"] == conversation_id and snapshot["status"] == "need_input":
            saved = snapshot["state"]
            original = saved.get("original_input") or saved.get("user_input") or ""
            if not original:
                return jsonify(error="待补充任务数据不完整，请重新发起求职需求。"), 409
            # 进程重启后 pending_tasks 为空，从 SQLite 快照恢复同样的澄清上下文。
            message = f"{original}\n补充信息：{message}"
            resume = saved.get("resume_path") or resume
            resume_filename = saved.get("resume_filename") or resume_filename
            task_id = ""
        elif not snapshot or snapshot["conversation_id"] != conversation_id:
            return jsonify(error="任务不存在或不属于当前会话。"), 404

    return Response(
        stream_with_context(
            stream_chat(
                message,
                resume,
                task_id,
                conversation_id,
                resume_filename,
                conversation_history,
            )
        ),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


atexit.register(browser.close)
atexit.register(vector_store.close)
atexit.register(conversation_store.close)
atexit.register(close_db)
# 退出处理按注册逆序执行，因此监听器必须在浏览器关闭前停止。
atexit.register(chat_worker.stop)

# 本模块只负责定义 Flask 应用与共享实例，不提供直接运行入口。
# 启动服务请使用 python main.py：它会在启动时读取 auto_reply 配置并拉起聊天监听线程，
# 而直接运行本模块会跳过该步骤，导致自动回复静默失效。
