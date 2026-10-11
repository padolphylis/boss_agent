import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Empty, Full, Queue
from random import uniform
from threading import Event, Lock, RLock, Thread

from browser import (
    AccessRestricted,
    BrowserManager,
    BrowserOperationCancelled,
    DeliveryRateLimited,
    LoginRequired,
    VerificationRequired,
    delivery_rate_limit_messages,
)
from config import get as cfg
from delivery_store import DeliveryStore
from logging_config import get_logger
import matcher
from matcher import code_book, match_card_batch
from analysis_work_content import full_recall_threshold
from semantic_match import SemanticMatchError, semantic_concurrency
from task_control import TaskCancelled, TaskControl

logger = get_logger(__name__)

default_delivery_batch_size = 6
default_delivery_delay_range = (4.0, 8.0)
delivery_serial_lock = Lock()
delivery_pacing_lock = Lock()
last_delivery_started_at: float | None = None


def _error_source(exc: Exception, default: str = "") -> str:
    """从异常链提取可供前端展示的错误来源。"""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        source = getattr(current, "error_source", "")
        if source:
            return str(source)
        current = current.__cause__ or current.__context__
    return default


def delivery_batch_size() -> int:
    """返回一次持有浏览器锁的最大投递数量。"""
    try:
        configured = cfg("delivery_batch_size").strip()
    except (AttributeError, TypeError, ValueError):
        configured = ""
    if configured:
        try:
            return max(int(configured), 1)
        except (TypeError, ValueError):
            logger.warning(
                "delivery_batch_size 配置无效，改用默认值: %s",
                configured,
            )
    return default_delivery_batch_size


def delivery_delay_range() -> tuple[float, float]:
    """返回职位间投递间隔，避免连续请求触发平台频率限制。"""
    try:
        minimum = float(cfg("delivery_delay_min_seconds") or default_delivery_delay_range[0])
        maximum = float(cfg("delivery_delay_max_seconds") or default_delivery_delay_range[1])
    except (TypeError, ValueError):
        minimum, maximum = default_delivery_delay_range
    minimum = max(minimum, 0.0)
    maximum = max(maximum, minimum)
    return minimum, maximum


def _build_search_plan(search_params):
    """展开 BOSS 只能单选的职位、城市、求职类型和薪资条件。"""
    raw_positions = getattr(search_params, "zhi_wei", None)
    positions = (
        [raw_positions]
        if isinstance(raw_positions, str)
        else list(raw_positions or [])
    )
    if not positions:
        positions = [""]
    locations = list(search_params.location)
    if not locations:
        locations = ["全国"] if search_params.location_unlimited else [""]
    job_types = list(search_params.job_type) or [None]
    salaries = list(search_params.money) or [None]
    return [
        {
            "position": position,
            "city": city,
            "job_type": job_type,
            "money": money,
        }
        for position in positions
        for city in locations
        for job_type in job_types
        for money in salaries
    ]


def _search_task_label(search_task: dict) -> str:
    """生成可展示的搜索子任务描述，避免日志和提示重复拼接条件。"""
    return " / ".join([
        search_task.get("position") or "不限职位",
        search_task.get("city") or "当前城市",
        search_task.get("job_type") or "不限类型",
        search_task.get("money") or "不限薪资",
    ])


def _format_search_failures(failures: list[dict], total: int) -> str:
    """把可恢复的子任务失败压缩成用户可理解的降级提示。"""
    if not failures:
        return ""
    labels = ", ".join(item["label"] for item in failures[:3])
    if len(failures) > 3:
        labels += f" 等 {len(failures)} 个组合"
    return (
        f"搜索计划共 {total} 个组合，其中 {len(failures)} 个组合未完成"
        f"（{labels}），已继续处理其余组合。"
    )


def _filter_matched_by_hard_filters(matched, search_params):
    """过滤旧快照或其他调用方传入的匹配结果，避免绕过硬条件。"""
    return [
        item
        for item in matched
        if matcher.card_passes_hard_filters(
            item.get("job_card") or item,
            getattr(search_params, "exclude_keywords", []),
            search_params,
        )
    ]


def match_job_content(state) -> dict:
    """对详情卡片执行排除过滤和批量向量匹配。"""
    if state.matched_jobs:
        matched = _filter_matched_by_hard_filters(
            state.matched_jobs,
            state.search_params,
        )
        return {
            "matched_jobs": matched,
            "result": f"排除后匹配到 {len(matched)} 个职位",
            "status": "matched",
        }
    cards = state.job_cards
    if not cards:
        return {"matched_jobs": [], "status": "no_cards", "working": False}
    try:
        excluded_keywords = state.search_params.exclude_keywords
        match_query = (
            state.retrieval_query
            or state.match_query
            or state.user_input
        )
        matched = match_card_batch(
            cards,
            match_query,
            excluded_keywords,
            search_params=state.search_params,
            threshold=full_recall_threshold,
        )
        eligible_count = sum(
            matcher.card_passes_hard_filters(
                card,
                excluded_keywords,
                state.search_params,
            )
            for card in cards
        )
        if not eligible_count:
            return {
                "matched_jobs": [],
                "result": f"排除不符合的工作内容后剩余 0/{len(cards)} 个职位",
                "status": "matched",
            }
        return {
            "matched_jobs": matched,
            "result": f"排除后匹配到 {len(matched)}/{eligible_count} 个职位",
            "status": "matched",
        }
    except TaskCancelled:
        raise
    except Exception as exc:
        logger.exception("向量匹配失败: task_id=%s", state.task_id)
        return {
            "error": f"Embedding 匹配失败: {type(exc).__name__}: {exc}",
            "error_source": _error_source(exc, "embedding"),
            "status": "match_failed",
            "working": False,
        }


def push_jobs(
    browser,
    state,
    browser_io_lock,
    task_control: TaskControl | None = None,
) -> dict:
    """向匹配职位投递简历。"""
    if state.delivery_processed:
        return {
            "status": state.status,
            "working": False,
            "pipeline_processed": True,
        }
    if state.pipeline_processed and state.push_results:
        return {"status": state.status, "working": state.working}
    if state.status == "search_timeout":
        timeout_error = (
            state.error
            or "职位处理超时，后台生产和匹配线程已停止；本次未继续投递。"
        )
        return {
            "push_results": [],
            "result": (
                state.pipeline_warning
                or timeout_error
            ),
            "error": timeout_error,
            "status": "search_timeout",
            "working": False,
            "pipeline_processed": True,
        }
    if not state.semantic_evaluated:
        return {
            "push_results": [],
            "result": "语义匹配尚未完成，已阻止投递。",
            "error": "语义匹配尚未完成，不能投递。",
            "status": "semantic_required",
            "working": False,
            "pipeline_processed": True,
        }
    matched = _filter_matched_by_hard_filters(
        state.matched_jobs,
        state.search_params,
    )
    if not matched:
        result = (
            f"找到 {len(state.jobs)} 个职位，获取 {len(state.job_cards)} 个详情，"
            "没有可投递的匹配职位"
        )
        if state.pipeline_warning:
            result = f"{state.pipeline_warning}\n{result}"
        return {
            "push_results": [],
            "result": result,
            "status": "completed",
            "working": False,
            "pipeline_processed": True,
        }
    try:
        results = push_matches(
            browser,
            matched,
            browser_io_lock,
            state.task_id,
            task_control,
            continue_on_error=True,
        )
        ok = sum(1 for result in results if result.get("success"))
        result = (
            f"找到 {len(state.jobs)} 个职位，获取 {len(state.job_cards)} 个详情，"
            f"匹配 {len(matched)} 个，投递成功 {ok} 个"
        )
        if state.pipeline_warning:
            result = f"{state.pipeline_warning}\n{result}"
        logger.info("职位投递完成: task_id=%s success=%s total=%s", state.task_id, ok, len(matched))
        return {
            "push_results": results,
            "result": result,
            "status": "completed",
            "working": False,
            "error": "",
            "pipeline_processed": True,
        }
    except Exception as exc:
        if isinstance(exc, TaskCancelled):
            raise
        logger.exception("职位投递失败: task_id=%s", state.task_id)
        if isinstance(exc, DeliveryRateLimited):
            partial_results = getattr(exc, "partial_results", [])
            return {
                "push_results": partial_results,
                "error": "Boss 平台投递频率受限，已停止后续投递请求。",
                "error_source": "browser",
                "result": (
                    f"已记录 {len(partial_results)} 个投递结果；"
                    "Boss 平台已限流，后续职位未再投递。"
                ),
                "status": "delivery_rate_limited",
                "working": False,
                "delivery_processed": True,
                "pipeline_processed": True,
            }
        return {
            "push_results": getattr(exc, "partial_results", []),
            "error": f"投递失败: {type(exc).__name__}: {exc}",
            "status": "push_failed",
            "working": False,
        }

detail_queue_size = 16
_pipeline_sentinel = object()
_detail_flush_sentinel = object()
_delivery_flush_sentinel = object()


def match_batch_size() -> int:
    """返回单个详情批次大小。"""
    try:
        configured = cfg("match_batch_size").strip()
    except (TypeError, ValueError):
        configured = ""
    if configured:
        try:
            return max(int(configured), 1)
        except (TypeError, ValueError):
            logger.warning("match_batch_size 配置无效，改用语义并发: %s", configured)
    return semantic_concurrency()


class _PipelineCancelled(Exception):
    """消费者失败后通知浏览器生产者尽快停止。"""


global_search_stop_errors = (
    LoginRequired,
    VerificationRequired,
    AccessRestricted,
    BrowserOperationCancelled,
    TaskCancelled,
    _PipelineCancelled,
)


def _queue_put(detail_queue, item, stop_event, consumer_done, *, force=False):
    while True:
        if consumer_done.is_set() or (stop_event.is_set() and not force):
            raise _PipelineCancelled
        try:
            detail_queue.put(item, timeout=0.1)
            return
        except Full:
            if consumer_done.is_set() or (stop_event.is_set() and not force):
                raise _PipelineCancelled


def _wait_for_queue(queue, stop_event, worker_done, task_control=None):
    """等待已入队工作完成；消费者退出或取消时不再无限等待 join。"""
    while True:
        if task_control is not None:
            task_control.checkpoint()
        if stop_event.is_set():
            raise _PipelineCancelled
        with queue.all_tasks_done:
            if not queue.unfinished_tasks:
                return
            if worker_done.is_set():
                raise _PipelineCancelled
            queue.all_tasks_done.wait(0.1)


def _browser_call_with_stop(
    browser: BrowserManager,
    method_name: str,
    *args,
    should_stop: Callable[[], bool],
    **kwargs,
):
    """调用可取消的浏览器接口，同时兼容旧的测试替身。"""
    method = getattr(browser, method_name)
    if type(method).__module__ == "unittest.mock":
        return method(*args, **kwargs)
    if isinstance(browser, BrowserManager):
        return method(*args, should_stop=should_stop, **kwargs)
    return method(*args, **kwargs)


@contextmanager
def _delivery_batch_lock(
    browser_io_lock: RLock,
    task_control: TaskControl | None,
) -> Iterator[Callable[[], None]]:
    """批次内复用锁；暂停先释放锁，恢复后重新获取再继续当前职位。"""
    acquired = False
    interrupted_states = {"pause_requested", "paused", "cancel_requested"}

    def acquire() -> None:
        nonlocal acquired
        while True:
            if task_control is not None:
                task_control.checkpoint()
            acquired = browser_io_lock.acquire(timeout=0.1)
            if not acquired:
                continue
            if (
                task_control is None
                or task_control.status not in interrupted_states
            ):
                return
            browser_io_lock.release()
            acquired = False

    def checkpoint() -> None:
        nonlocal acquired
        if (
            task_control is not None
            and task_control.status in interrupted_states
        ):
            browser_io_lock.release()
            acquired = False
            acquire()

    try:
        acquire()
        yield checkpoint
    finally:
        if acquired:
            browser_io_lock.release()


def _wait_for_delivery_slot(
    checkpoint: Callable[[], object] | None = None,
) -> None:
    """为下一次投递请求分配全局时间槽。"""
    global last_delivery_started_at

    delay = uniform(*delivery_delay_range())
    while True:
        if checkpoint is not None:
            checkpoint()
        with delivery_pacing_lock:
            now = time.monotonic()
            wait_seconds = (
                0.0
                if last_delivery_started_at is None
                else max(
                    0.0,
                    last_delivery_started_at + delay - now,
                )
            )
            if wait_seconds <= 0:
                last_delivery_started_at = now
                return
        interval = min(wait_seconds, 0.2)
        before_sleep = time.monotonic()
        time.sleep(interval)
        if time.monotonic() - before_sleep < 0.001:
            return


def push_matches(
    browser: BrowserManager,
    matched: list[dict],
    browser_io_lock: RLock,
    task_id: str = "",
    task_control: TaskControl | None = None,
    *,
    start_index: int = 0,
    store: DeliveryStore | None = None,
    batch_size: int | None = None,
    continue_on_error: bool = False,
) -> list[dict]:
    """按批次持有浏览器锁完成投递，并持久化职位投递状态。"""
    global last_delivery_started_at
    try:
        requested_batch_size = (
            delivery_batch_size()
            if batch_size is None
            else int(batch_size)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("delivery batch_size 必须是正整数") from exc
    if requested_batch_size <= 0:
        raise ValueError("delivery batch_size 必须是正整数")
    if not matched:
        return []

    results = []
    owns_store = store is None
    store = store if store is not None else DeliveryStore()
    serial_lock_acquired = False
    try:
        while True:
            if task_control is not None:
                task_control.checkpoint()
            serial_lock_acquired = delivery_serial_lock.acquire(timeout=0.1)
            if serial_lock_acquired:
                break
        for batch_start in range(0, len(matched), requested_batch_size):
            batch = matched[batch_start:batch_start + requested_batch_size]
            if task_control is not None:
                task_control.checkpoint()
            with _delivery_batch_lock(
                browser_io_lock, task_control
            ) as checkpoint:
                logger.info(
                    "开始投递批次: task_id=%s start=%s size=%s total=%s",
                    task_id,
                    start_index + batch_start + 1,
                    len(batch),
                    len(matched),
                )
                for offset, item in enumerate(batch):
                    checkpoint()
                    index = start_index + batch_start + offset + 1
                    card = item.get("job_card") or item
                    previous = store.get(str(card.get("encryptJobId") or ""))
                    if previous and previous.get("status") == "success":
                        results.append({
                            "job_card": card,
                            "success": True,
                            "status": "success",
                            "message": "已跳过：此前状态为 success",
                            "skipped": True,
                        })
                        continue
                    title = card.get("jobName") or (
                        card.get("postDescription") or ""
                    )[:20]
                    if browser.check_yan_cheng_ma():
                        message = "验证码阻断"
                        results.append(
                            {
                                "job_card": card,
                                "success": False,
                                "status": "failed",
                                "message": message,
                            }
                        )
                        raise VerificationRequired("投递时检测到验证码")
                    _wait_for_delivery_slot(checkpoint)
                    checkpoint()
                    reserved, previous_status = store.reserve(card, task_id)
                    if not reserved:
                        message = f"已跳过：此前状态为 {previous_status}"
                        results.append(
                            {
                                "job_card": card,
                                "success": previous_status == "success",
                                "status": previous_status,
                                "message": message,
                                "skipped": True,
                            }
                        )
                        continue

                    try:
                        with delivery_pacing_lock:
                            last_delivery_started_at = time.monotonic()
                        ok, message = browser.push_job(card)
                    except Exception as exc:
                        message = f"结果未知：{type(exc).__name__}: {exc}"
                        results.append(
                            {
                                "job_card": card,
                                "success": False,
                                "status": "unknown",
                                "message": message,
                            }
                        )
                        store.update(card, "unknown", message, task_id)
                        if not continue_on_error or isinstance(
                            exc,
                            (
                                TaskCancelled,
                                LoginRequired,
                                VerificationRequired,
                                AccessRestricted,
                            ),
                        ):
                            raise
                        logger.exception(
                            "单职位投递异常，继续当前投递批次: "
                            "task_id=%s job_id=%s",
                            task_id,
                            card.get("encryptJobId", ""),
                        )
                        continue

                    status = "success" if ok else (
                        "unknown"
                        if str(message).startswith("结果未知：")
                        else "failed"
                    )
                    results.append(
                        {
                            "job_card": card,
                            "success": ok,
                            "status": status,
                            "message": message,
                        }
                    )
                    store.update(card, status, message, task_id)
                    logger.info(
                        "职位投递%s: index=%s total=%s title=%s message=%s",
                        "成功" if ok else "失败",
                        index,
                        len(matched),
                        title,
                        message,
                    )
                    if (
                        not ok
                        and any(
                            marker in str(message)
                            for marker in delivery_rate_limit_messages
                        )
                    ):
                        blocked = DeliveryRateLimited(
                            f"Boss 投递频率受限：{message}"
                        )
                        blocked.partial_results = list(results)
                        raise blocked
            if task_control is not None:
                task_control.checkpoint()
        return results
    except Exception as exc:
        exc.partial_results = list(results)
        raise
    finally:
        if owns_store:
            store.close()
        if serial_lock_acquired:
            delivery_serial_lock.release()


def push_single_match(
    browser: BrowserManager,
    matched: dict,
    browser_io_lock: RLock,
    task_id: str = "",
    task_control: TaskControl | None = None,
    *,
    start_index: int = 0,
    store: DeliveryStore | None = None,
) -> dict:
    """投递一个已经完成语义审核的职位。"""
    results = push_matches(
        browser,
        [matched],
        browser_io_lock,
        task_id,
        task_control,
        start_index=start_index,
        store=store,
    )
    if not results:
        raise RuntimeError("单职位投递没有返回结果")
    return results[0]


def search_jobs(
    browser: BrowserManager,
    state,
    browser_io_lock: RLock,
    task_control: TaskControl | None = None,
    on_batch_matches: Callable[[list[dict], Event, DeliveryStore | None], dict | None] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """抓取职位详情并按批次匹配。"""
    with browser_io_lock:
        browser.ensure_or_wait()
        if not browser.check_login():
            logger.warning(
                "职位读取前再次确认登录失败，返回登录等待: task_id=%s",
                state.task_id,
            )
            return {
                "error": "",
                "status": "login_required",
                "working": False,
            }

    p = state.search_params
    search_plan = _build_search_plan(p)
    batch_size = match_batch_size()
    excluded = {c.strip().rstrip("市") for c in p.exclude_location}
    jobs_by_key = {}
    cards = []
    pipeline_data_lock = Lock()
    detail_queue = Queue(maxsize=detail_queue_size)
    producer_stop_event = Event()
    consumer_cancel_event = Event()
    consumer_done = Event()
    producer_done = Event()
    producer_error = []
    consumer_error = []
    delivery_error = []
    search_failures = []
    matched = []
    semantic_results = []
    push_results = []
    matched_lock = Lock()
    semantic_results_lock = Lock()
    push_results_lock = Lock()
    delivery_queue = Queue()
    delivery_done = Event()
    delivery_blocked = Event()
    delivery_enabled = (
        on_batch_matches is not None
        and state.intent != "job_recommendation"
    )
    delivery_keys: set[str] = set()
    delivery_store: DeliveryStore | None = None
    progress_lock = RLock()
    last_progress_at = 0.0
    current_search = {
        "index": 0,
        "total": len(search_plan),
        "label": "",
        "page": 0,
    }
    match_query = (
        state.retrieval_query
        or state.match_query
        or state.user_input
    )

    def pipeline_snapshot():
        with pipeline_data_lock:
            return list(jobs_by_key.values()), list(cards)

    def emit_progress(
        phase: str,
        *,
        force: bool = False,
        page: int | None = None,
    ) -> None:
        """把管线安全检查点同步给任务快照和前端。"""
        with progress_lock:
            publish_progress(phase, force=force, page=page)

    def publish_progress(phase, *, force=False, page=None):
        nonlocal last_progress_at
        if on_progress is None:
            return

        now = time.monotonic()
        with progress_lock:
            if not force and now - last_progress_at < 1.5:
                return
            last_progress_at = now
            if page is not None:
                current_search["page"] = page
            search_index = current_search["index"]
            search_total = current_search["total"]
            search_label = current_search["label"]
            current_page = current_search["page"]

        with pipeline_data_lock:
            jobs_count = len(jobs_by_key)
            details_count = len(cards)
        with matched_lock:
            matched_count = len(matched)
        with semantic_results_lock:
            semantic_checked_count = len(semantic_results)
        with push_results_lock:
            delivery_success_count = sum(
                1
                for item in push_results
                if item.get("success") and not item.get("skipped")
            )
            delivery_skipped_count = sum(
                1 for item in push_results if item.get("skipped")
            )
        try:
            on_progress(
                {
                    "semantic_evaluated": False,
                    "delivery_processed": bool(delivery_enabled),
                    "pipeline_processed": False,
                    "status": "searching",
                    "working": True,
                    "checkpoint_node": "search_jobs",
                    "pipeline_progress": {
                        "phase": phase,
                        "search_index": search_index,
                        "search_total": search_total,
                        "search_label": search_label,
                        "page": current_page,
                        "jobs": jobs_count,
                        "details": details_count,
                        "semantic_checked": semantic_checked_count,
                        "matched": matched_count,
                        "delivery_success": delivery_success_count,
                        "delivery_skipped": delivery_skipped_count,
                        "delivery_blocked": delivery_blocked.is_set(),
                    },
                }
            )
        except Exception:
            logger.warning(
                "任务进度快照更新失败，继续执行职位管线: task_id=%s",
                state.task_id,
                exc_info=True,
            )

    def delivery_key(item: dict) -> str:
        card = item.get("job_card") or item
        return str(
            card.get("encryptJobId")
            or card.get("jobId")
            or "|".join(
                str(card.get(field) or "")
                for field in ("jobName", "brandName", "salaryDesc")
            )
        )

    def enqueue_delivery(items: list[dict]) -> None:
        """把语义通过的职位交给独立投递线程，避免阻塞详情消费者。"""
        if not delivery_enabled or delivery_blocked.is_set():
            return
        for item in items:
            if producer_stop_event.is_set():
                return
            key = delivery_key(item)
            if key in delivery_keys:
                continue
            delivery_keys.add(key)
            delivery_queue.put(item)

    def deliver_batch(items: list[dict], store: DeliveryStore) -> None:
        if (
            not items
            or consumer_cancel_event.is_set()
            or delivery_blocked.is_set()
        ):
            return
        try:
            results = push_matches(
                browser,
                items,
                browser_io_lock,
                state.task_id,
                task_control,
                store=store,
                batch_size=delivery_batch_size(),
                continue_on_error=True,
            )
        except DeliveryRateLimited as exc:
            with push_results_lock:
                push_results.extend(getattr(exc, "partial_results", []))
            delivery_error.append(exc)
            delivery_blocked.set()
            emit_progress("delivery_rate_limited", force=True)
            logger.warning(
                "Boss 投递被限流，停止后续投递但继续完成搜索: task_id=%s",
                state.task_id,
            )
            return
        except Exception as exc:
            partial_results = getattr(exc, "partial_results", [])
            with push_results_lock:
                push_results.extend(partial_results)
            raise
        with push_results_lock:
            push_results.extend(results)
        emit_progress("delivery", force=True)

    def consume_deliveries() -> None:
        """在搜索仍进行时消费语义通过结果，按小批次串行投递。"""
        nonlocal delivery_store
        if not delivery_enabled:
            delivery_done.set()
            return

        pending = []
        pending_queue_items = []

        def clear_pending() -> None:
            while pending_queue_items:
                delivery_queue.task_done()
                pending_queue_items.pop()
            pending.clear()

        try:
            delivery_store = DeliveryStore()
            while True:
                item = delivery_queue.get()
                if item is _pipeline_sentinel:
                    try:
                        if (
                            pending
                            and not consumer_cancel_event.is_set()
                            and not delivery_blocked.is_set()
                        ):
                            deliver_batch(pending, delivery_store)
                    finally:
                        clear_pending()
                        delivery_queue.task_done()
                    break
                if item is _delivery_flush_sentinel:
                    try:
                        if (
                            pending
                            and not consumer_cancel_event.is_set()
                            and not delivery_blocked.is_set()
                        ):
                            deliver_batch(pending, delivery_store)
                    finally:
                        clear_pending()
                        delivery_queue.task_done()
                    continue
                if consumer_cancel_event.is_set() or delivery_blocked.is_set():
                    delivery_queue.task_done()
                    continue
                pending.append(item)
                pending_queue_items.append(item)
                if len(pending) >= delivery_batch_size():
                    deliver_batch(pending, delivery_store)
                    clear_pending()
            if pending and not consumer_cancel_event.is_set():
                deliver_batch(pending, delivery_store)
            clear_pending()

        except TaskCancelled as exc:
            delivery_error.append(exc)
            producer_stop_event.set()
            consumer_cancel_event.set()
        except Exception as exc:
            exc.partial_results = list(push_results)
            delivery_error.append(exc)
            producer_stop_event.set()
            consumer_cancel_event.set()
            logger.exception(
                "独立投递消费者停止: task_id=%s error=%s",
                state.task_id,
                type(exc).__name__,
            )
        finally:
            clear_pending()
            while True:
                try:
                    delivery_queue.get_nowait()
                except Empty:
                    break
                else:
                    delivery_queue.task_done()
            if delivery_store is not None:
                delivery_store.close()
                delivery_store = None
            delivery_done.set()

    def fetch_detail(job):
        if task_control is not None:
            task_control.checkpoint()
        if producer_stop_event.is_set():
            raise _PipelineCancelled
        city_name = str(job.get("cityName") or "").strip().rstrip("市")
        if city_name and city_name in excluded:
            return
        if not matcher.card_matches_boss_activity(
            job,
            p,
            allow_unknown=p.boss_activity == "recent",
        ):
            return
        if not matcher.card_passes_hard_filters(
            job,
            p.exclude_keywords,
            p,
            allow_unknown_activity=p.boss_activity == "recent",
            allow_unknown_salary=bool(p.salary),
        ):
            return
        key = browser.job_key(job)
        with pipeline_data_lock:
            if key in jobs_by_key:
                return
            jobs_by_key[key] = job
        try:
            card = _browser_call_with_stop(
                browser,
                "get_job_card",
                job,
                should_stop=producer_stop_event.is_set,
            )
            if task_control is not None:
                task_control.checkpoint()
            if producer_stop_event.is_set():
                with pipeline_data_lock:
                    jobs_by_key.pop(key, None)
                raise _PipelineCancelled
        except (VerificationRequired, AccessRestricted) as exc:
            _, partial_cards = pipeline_snapshot()
            exc.partial_cards = partial_cards
            raise
        except global_search_stop_errors:
            with pipeline_data_lock:
                jobs_by_key.pop(key, None)
            raise
        except Exception as exc:
            with pipeline_data_lock:
                jobs_by_key.pop(key, None)
            logger.warning(
                "职位详情读取失败，跳过当前职位: task_id=%s title=%s error=%s",
                state.task_id,
                job.get("jobName", ""),
                type(exc).__name__,
            )
            return
        if card is not None:
            if (
                not matcher.card_passes_hard_filters(
                    card,
                    p.exclude_keywords,
                    p,
                )
            ):
                with pipeline_data_lock:
                    jobs_by_key.pop(key, None)
                return
            with pipeline_data_lock:
                cards.append(card)
            _queue_put(detail_queue, card, producer_stop_event, consumer_done)
            emit_progress("details")
        elif p.boss_activity == "recent":
            with pipeline_data_lock:
                jobs_by_key.pop(key, None)

    def consume_details():
        batch = []

        def consume_batch(items):
            if not items:
                return
            if task_control is not None:
                task_control.checkpoint()
            batch_matches = match_card_batch(
                items,
                match_query,
                p.exclude_keywords,
                search_params=p,
                threshold=full_recall_threshold,
            )
            if consumer_cancel_event.is_set():
                return
            if on_batch_matches is None:
                with matched_lock:
                    matched.extend(batch_matches)
                return

            try:
                outcome = on_batch_matches(
                    batch_matches,
                    consumer_cancel_event,
                    None,
                ) or {}
            except Exception as exc:
                partial_accepted = list(
                    getattr(exc, "partial_accepted", [])
                )
                enqueue_delivery(partial_accepted)
                raise
            accepted_jobs = outcome.get("accepted_jobs") or []
            with matched_lock:
                matched.extend(accepted_jobs)
            with semantic_results_lock:
                semantic_results.extend(outcome.get("semantic_results") or [])
            enqueue_delivery(accepted_jobs)
            emit_progress("semantic", force=True)

        try:
            while True:
                try:
                    item = detail_queue.get(timeout=0.1)
                except Empty:
                    if producer_done.is_set():
                        break
                    continue
                try:
                    if item is _pipeline_sentinel:
                        break
                    if item is _detail_flush_sentinel:
                        if batch and not consumer_cancel_event.is_set():
                            consume_batch(batch)
                        batch.clear()
                        continue
                    if consumer_cancel_event.is_set():
                        break
                    batch.append(item)
                    if len(batch) >= batch_size:
                        consume_batch(batch)
                        batch.clear()
                finally:
                    detail_queue.task_done()
            if batch and not consumer_cancel_event.is_set():
                consume_batch(batch)
        except TaskCancelled as exc:
            consumer_error.append(exc)
            consumer_cancel_event.set()
            producer_stop_event.set()
        except Exception as exc:
            consumer_error.append(exc)
            consumer_cancel_event.set()
            producer_stop_event.set()
        finally:
            consumer_done.set()
            if delivery_enabled:
                delivery_queue.put(_pipeline_sentinel)

    def produce_details():
        try:
            for task_index, search_task in enumerate(search_plan, 1):
                if task_control is not None:
                    task_control.checkpoint()
                if producer_stop_event.is_set():
                    raise _PipelineCancelled
                task_label = _search_task_label(search_task)
                with progress_lock:
                    current_search["index"] = task_index
                    current_search["label"] = task_label
                    current_search["page"] = 0
                emit_progress("searching", force=True)
                logger.info(
                    "开始职位搜索子任务: task_id=%s index=%s/%s position=%s city=%s job_type=%s money=%s",
                    state.task_id,
                    task_index,
                    len(search_plan),
                    search_task["position"] or "不限",
                    search_task["city"] or "当前城市",
                    search_task["job_type"] or "不限",
                    search_task["money"] or "不限",
                )
                try:
                    def on_page_progress(page_info: dict) -> None:
                        emit_progress(
                            "listing",
                            force=True,
                            page=int(page_info.get("page") or 0),
                        )

                    with browser_io_lock:
                        jobs = _browser_call_with_stop(
                            browser,
                            "get_job_list",
                            search_task["position"],
                            city=search_task["city"],
                            salary_code=code_book.code_of(
                                "salary",
                                search_task["money"] or "",
                            ),
                            experience_code=code_book.codes_of("experience", p.experience),
                            scale_code=code_book.codes_of("scale", p.scale),
                            degree_code=code_book.codes_of("degree", p.degree),
                            job_type_code=code_book.code_of(
                                "job_type",
                                search_task["job_type"] or "",
                            ),
                            stage_code=code_book.codes_of("stage", p.stage),
                            on_job=fetch_detail,
                            on_page=on_page_progress,
                            should_stop=producer_stop_event.is_set,
                        )
                    for job in jobs:
                        if producer_stop_event.is_set():
                            raise _PipelineCancelled
                        city_name = str(job.get("cityName") or "").strip().rstrip("市")
                        if city_name and city_name in excluded:
                            continue
                        if not matcher.card_passes_hard_filters(
                            job,
                            p.exclude_keywords,
                            p,
                            allow_unknown_salary=bool(p.salary),
                        ):
                            continue
                        with pipeline_data_lock:
                            jobs_by_key.setdefault(browser.job_key(job), job)
                    if delivery_enabled and not producer_stop_event.is_set():
                        _queue_put(
                            detail_queue,
                            _detail_flush_sentinel,
                            producer_stop_event,
                            consumer_done,
                        )
                        _wait_for_queue(
                            detail_queue,
                            producer_stop_event,
                            consumer_done,
                            task_control,
                        )
                        if delivery_enabled and not delivery_done.is_set():
                            delivery_queue.put(_delivery_flush_sentinel)
                            _wait_for_queue(
                                delivery_queue,
                                producer_stop_event,
                                delivery_done,
                                task_control,
                            )
                    emit_progress("searching", force=True)
                except global_search_stop_errors:
                    raise
                except Exception as exc:
                    failure = {
                        "index": task_index,
                        "label": task_label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    search_failures.append(failure)
                    logger.exception(
                        "职位搜索子任务失败，继续后续组合: task_id=%s index=%s/%s label=%s",
                        state.task_id,
                        task_index,
                        len(search_plan),
                        task_label,
                    )
        except TaskCancelled as exc:
            producer_error.append(exc)
            producer_stop_event.set()
        except _PipelineCancelled:
            if (
                not producer_stop_event.is_set()
                and not consumer_error
            ):
                producer_error.append(_PipelineCancelled())
        except Exception as exc:
            if not (
                isinstance(exc, BrowserOperationCancelled)
                and producer_stop_event.is_set()
            ):
                producer_error.append(exc)
                producer_stop_event.set()
        finally:
            if not consumer_done.is_set():
                try:
                    _queue_put(
                        detail_queue,
                        _pipeline_sentinel,
                        producer_stop_event,
                        consumer_done,
                        force=True,
                    )
                except _PipelineCancelled:
                    pass
            producer_done.set()

    consumer = Thread(target=consume_details, name=f"job-match-{state.task_id[:8]}", daemon=True)
    producer = Thread(target=produce_details, name=f"job-fetch-{state.task_id[:8]}", daemon=True)
    delivery = Thread(target=consume_deliveries, name=f"job-push-{state.task_id[:8]}", daemon=True)
    consumer.start()
    delivery.start()
    producer.start()

    try:
        while (
            producer.is_alive()
            or consumer.is_alive()
            or delivery.is_alive()
        ):
            if task_control is not None:
                task_control.checkpoint()

            producer.join(0.1)
            consumer.join(0.1)
            delivery.join(0.1)

    except TaskCancelled:
        producer_stop_event.set()
        consumer_cancel_event.set()
        producer.join()
        consumer.join()
        if delivery_enabled:
            delivery_queue.put(_pipeline_sentinel)
        delivery.join()
        raise

    if delivery_enabled and not delivery_done.is_set():
        delivery_queue.put(_pipeline_sentinel)
        delivery.join()

    if any(
        isinstance(exc, TaskCancelled)
        for exc in producer_error + consumer_error + delivery_error
    ):
        raise TaskCancelled

    if task_control is not None:
        task_control.checkpoint()

    with push_results_lock:
        push_results_snapshot = list(push_results)

    with matched_lock:
        matched_snapshot = list(matched)
    jobs_snapshot, cards_snapshot = pipeline_snapshot()

    if delivery_error:
        exc = delivery_error[0]
        if isinstance(exc, DeliveryRateLimited):
            return {
                "jobs": jobs_snapshot,
                "job_cards": cards_snapshot,
                "matched_jobs": matched_snapshot,
                "semantic_results": list(semantic_results),
                "push_results": push_results_snapshot,
                "semantic_evaluated": bool(on_batch_matches),
                "delivery_processed": True,
                "pipeline_processed": True,
                "result": (
                    f"已记录 {len(push_results_snapshot)} 个投递结果；"
                    "Boss 平台已限流，后续职位未再投递。"
                ),
                "error": "",
                "error_source": "browser",
                "status": "search_completed",
                "working": False,
                "pipeline_warning": (
                    "Boss 平台触发投递频率限制，后续投递已停止；"
                    f"本次成功 {sum(1 for item in push_results_snapshot if item.get('success') and not item.get('skipped'))} 条，"
                    f"已保留 {len(matched_snapshot)} 条语义通过职位。"
                ),
            }

        if isinstance(exc, (LoginRequired, VerificationRequired, AccessRestricted)):
            error_source = "browser"
            error = f"投递被阻断: {type(exc).__name__}: {exc}"
        else:
            error_source = _error_source(exc, "browser")
            error = f"投递失败: {type(exc).__name__}: {exc}"
        return {
            "jobs": jobs_snapshot,
            "job_cards": cards_snapshot,
            "matched_jobs": matched_snapshot,
            "semantic_results": list(semantic_results),
            "push_results": push_results_snapshot,
            "semantic_evaluated": bool(on_batch_matches),
            "delivery_processed": True,
            "pipeline_processed": True,
            "result": (
                f"已记录 {len(push_results_snapshot)} 个投递结果，"
                "后续职位已停止投递。"
            ),
            "error": error,
            "error_source": error_source,
            "status": "push_failed",
            "working": False,
        }

    if producer_error and not consumer_error:
        exc = producer_error[0]
        if isinstance(exc, LoginRequired):
            logger.warning(
                "职位搜索与详情获取中断，返回登录等待: task_id=%s reason=%s",
                state.task_id,
                exc,
            )
            with matched_lock:
                matched_snapshot = list(matched)
            jobs_snapshot, cards_snapshot = pipeline_snapshot()
            return {
                "jobs": jobs_snapshot,
                "job_cards": cards_snapshot,
                "matched_jobs": matched_snapshot,
                "semantic_results": list(semantic_results),
                "push_results": list(push_results),
                "semantic_evaluated": False,
                "delivery_processed": bool(on_batch_matches),
                "result": "",
                "error": "",
                "status": "login_required",
                "working": False,
            }
        if isinstance(exc, (VerificationRequired, AccessRestricted)):
            partial_cards = getattr(exc, "partial_cards", None)
            cards = [card for card in (partial_cards or cards) if card is not None]
            logger.warning("职位搜索与详情获取中断: task_id=%s saved=%s reason=%s", state.task_id, len(cards), exc)
            with matched_lock:
                matched_snapshot = list(matched)
            jobs_snapshot, cards_snapshot = pipeline_snapshot()
            return {"jobs": jobs_snapshot, "job_cards": cards_snapshot, "matched_jobs": matched_snapshot,
                    "semantic_results": list(semantic_results),
                    "push_results": list(push_results),
                    "semantic_evaluated": False,
                    "delivery_processed": bool(on_batch_matches),
                    "result": f"职位读取已停止，保留 {len(cards)} 个职位详情",
                    "error": f"职位详情获取失败: {type(exc).__name__}: {exc}", "status": "cards_failed", "working": False}
        if not isinstance(exc, _PipelineCancelled):
            logger.error("职位搜索与详情获取失败: task_id=%s error=%s", state.task_id, f"{type(exc).__name__}: {exc}")
            with matched_lock:
                matched_snapshot = list(matched)
            jobs_snapshot, cards_snapshot = pipeline_snapshot()
            return {"jobs": jobs_snapshot, "job_cards": cards_snapshot, "matched_jobs": matched_snapshot,
                    "semantic_results": list(semantic_results),
                    "push_results": list(push_results),
                    "semantic_evaluated": False,
                    "delivery_processed": bool(on_batch_matches),
                    "result": f"职位读取已停止，保留 {len(cards)} 个职位详情",
                    "error": f"职位搜索或职位详情获取失败: {type(exc).__name__}: {exc}", "status": "search_failed", "working": False}

    if consumer_error:
        exc = consumer_error[0]
        logger.error("职位匹配失败: task_id=%s error=%s", state.task_id, f"{type(exc).__name__}: {exc}")
        with matched_lock:
            matched_snapshot = list(matched)
        jobs_snapshot, cards_snapshot = pipeline_snapshot()
        if isinstance(exc, (VerificationRequired, AccessRestricted)):
            status = "push_failed"
            error = f"投递被阻断: {type(exc).__name__}: {exc}"
            error_source = "browser"
            result = (
                f"已完成 {len(push_results)} 个职位投递，"
                "后续职位因平台限制停止。"
            )
        elif isinstance(exc, SemanticMatchError):
            status = "semantic_failed"
            error = f"LLM 语义匹配失败: {type(exc).__name__}: {exc}"
            error_source = "llm"
            result = "语义匹配服务失败，已停止后续职位处理。"
        else:
            status = "match_failed"
            error = f"Embedding 匹配失败: {type(exc).__name__}: {exc}"
            error_source = _error_source(exc, "embedding")
            result = f"职位处理已停止，保留 {len(cards)} 个职位详情"
        partial_results = getattr(exc, "partial_results", None)
        partial_audit = getattr(exc, "partial_audit", None)
        partial_accepted = getattr(exc, "partial_accepted", None)
        if partial_results:
            push_results = list(partial_results)
        if partial_audit:
            semantic_results = list(partial_audit)
        if partial_accepted:
            with matched_lock:
                matched.extend(partial_accepted)
                matched_snapshot = list(matched)
        if isinstance(exc, SemanticMatchError) and partial_accepted:
            return {
                "jobs": jobs_snapshot,
                "job_cards": cards_snapshot,
                "matched_jobs": matched_snapshot,
                "semantic_results": list(semantic_results),
                "push_results": list(push_results),
                "semantic_evaluated": bool(on_batch_matches),
                "delivery_processed": False,
                "pipeline_processed": True,
                "result": (
                    f"语义模型服务暂时不可用，已保留 "
                    f"{len(matched_snapshot)} 个已审核通过职位，"
                    "未审核职位未进入投递。"
                ),
                "error": "",
                "error_source": "llm",
                "status": "search_completed",
                "working": False,
                "pipeline_warning": (
                    "语义模型服务异常，后续职位审核已停止；"
                    "仅处理已经完成语义审核的职位。"
                ),
            }
        return {
            "jobs": jobs_snapshot,
            "job_cards": cards_snapshot,
            "matched_jobs": matched_snapshot,
            "semantic_results": list(semantic_results),
            "push_results": list(push_results),
            "semantic_evaluated": False,
            "delivery_processed": bool(on_batch_matches),
            "result": result,
            "error": error,
            "error_source": error_source,
            "status": status,
            "working": False,
        }

    matched_snapshot.sort(key=lambda item: item.get("score", 0), reverse=True)
    jobs = jobs_snapshot
    pipeline_warning = ""
    status = "search_completed"
    failure_warning = _format_search_failures(
        search_failures,
        len(search_plan),
    )
    if failure_warning:
        pipeline_warning = failure_warning
    logger.info(
        "职位读取与匹配完成: task_id=%s jobs=%s cards=%s matched=%s",
        state.task_id,
        len(jobs),
        len(cards),
        len(matched),
    )
    return {
        "jobs": jobs,
        "job_cards": cards_snapshot,
        "matched_jobs": matched_snapshot,
        "semantic_results": list(semantic_results),
        "push_results": push_results_snapshot,
        "semantic_evaluated": bool(on_batch_matches),
        "delivery_processed": delivery_enabled,
        "pipeline_processed": bool(on_batch_matches),
        "result": (
            f"{pipeline_warning}\n" if pipeline_warning else ""
        ) + (
            f"找到 {len(jobs)} 个职位，获取 {len(cards)} 个详情，"
            f"语义通过 {len(matched_snapshot)} 个，"
            f"投递成功 {sum(1 for item in push_results if item.get('success') and not item.get('skipped'))} 个"
            if on_batch_matches and state.intent != "job_recommendation"
            else f"找到 {len(jobs)} 个职位，获取 {len(cards)} 个详情，匹配 {len(matched_snapshot)} 个"
        ),
        "status": status,
        "pipeline_warning": pipeline_warning,
    }
