"""职位搜索、详情匹配和逐条投递管线。"""

from queue import Full, Queue
from random import uniform
from threading import Event, RLock, Thread
import time

from browser import AccessRestricted, BrowserManager, VerificationRequired
from delivery_store import DeliveryStore
from logging_config import get_logger
import matcher
from matcher import code_book, match_card_batch

logger = get_logger(__name__)


def match_job_content(state) -> dict:
    """对详情卡片执行排除过滤和批量向量匹配。"""
    if state.pipeline_processed:
        return {"status": state.status, "working": state.working}
    cards = state.job_cards
    if not cards:
        return {"matched_jobs": [], "status": "no_cards", "working": False}
    try:
        excluded_keywords = state.search_params.exclude_keywords
        matched = match_card_batch(
            cards,
            state.user_input,
            excluded_keywords,
            search_params=state.search_params,
        )
        eligible_count = sum(
            not matcher.card_matches_exclusions(card, excluded_keywords)
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
    except Exception as exc:
        logger.exception("向量匹配失败: task_id=%s", state.task_id)
        return {
            "error": f"向量匹配失败: {type(exc).__name__}: {exc}",
            "status": "match_failed",
            "working": False,
        }


def push_jobs(browser, state, browser_io_lock) -> dict:
    """向匹配职位投递简历。"""
    if state.pipeline_processed:
        return {"status": state.status, "working": state.working}
    matched = state.matched_jobs
    if not matched:
        return {"push_results": [], "status": "no_matched", "working": False}
    try:
        results = push_matches(browser, matched, browser_io_lock)
        ok = sum(1 for result in results if result.get("success"))
        logger.info("职位投递完成: task_id=%s success=%s total=%s", state.task_id, ok, len(matched))
        return {
            "push_results": results,
            "result": f"投递 {ok}/{len(matched)} 个职位",
            "status": "completed",
            "working": False,
            "error": "",
        }
    except Exception as exc:
        logger.exception("职位投递失败: task_id=%s", state.task_id)
        return {
            "push_results": getattr(exc, "partial_results", []),
            "error": f"投递失败: {type(exc).__name__}: {exc}",
            "status": "push_failed",
            "working": False,
        }

DETAIL_QUEUE_SIZE = 8
MATCH_BATCH_SIZE = 8
_PIPELINE_SENTINEL = object()


class _PipelineCancelled(Exception):
    """消费者失败后通知浏览器生产者尽快停止。"""


def _queue_put(detail_queue, item, stop_event, consumer_done, *, force=False):
    while True:
        try:
            detail_queue.put(item, timeout=0.1)
            return
        except Full:
            if consumer_done.is_set() or (stop_event.is_set() and not force):
                raise _PipelineCancelled


def push_matches(
    browser: BrowserManager,
    matched: list[dict],
    browser_io_lock: RLock,
    task_id: str = "",
) -> list[dict]:
    """在浏览器锁内逐个投递，并持久化职位投递状态。"""
    results = []
    store = DeliveryStore()
    try:
        for index, item in enumerate(matched, 1):
            card = item.get("job_card") or item
            title = card.get("jobName", "") or card.get("postDescription", "")[:20]
            reserved, previous_status = store.reserve(card, task_id)
            if not reserved:
                message = f"已跳过：此前状态为 {previous_status}"
                results.append({"job_card": card, "success": previous_status == "success", "message": message, "skipped": True})
                continue
            if index > 1:
                time.sleep(uniform(1, 2))
            with browser_io_lock:
                if browser.check_yan_cheng_ma():
                    logger.warning("投递时检测到验证码: index=%s total=%s", index, len(matched))
                    store.update(card, "failed", "验证码阻断", task_id)
                    results.append({"job_card": card, "success": False, "message": "验证码阻断"})
                    blocked = VerificationRequired("投递时检测到验证码")
                    blocked.partial_results = list(results)
                    raise blocked
                try:
                    ok, message = browser.push_job(card)
                except Exception as exc:
                    message = f"结果未知：{type(exc).__name__}: {exc}"
                    store.update(card, "unknown", message, task_id)
                    exc.partial_results = list(results)
                    raise
            status = "success" if ok else (
                "unknown" if str(message).startswith("结果未知：") else "failed"
            )
            store.update(card, status, message, task_id)
            logger.info(
                "职位投递%s: index=%s total=%s title=%s message=%s",
                "成功" if ok else "失败", index, len(matched), title, message,
            )
            results.append({"job_card": card, "success": ok, "message": message})
        return results
    finally:
        store.close()


def search_jobs(browser: BrowserManager, state, browser_io_lock: RLock) -> dict:
    """抓取详情、匹配并投递职位，返回原 search_jobs 的状态更新。"""
    p = state.search_params
    excluded = {c.strip().rstrip("市") for c in p.exclude_location}
    jobs_by_key = {}
    cards = []
    detail_queue = Queue(maxsize=DETAIL_QUEUE_SIZE)
    stop_event = Event()
    consumer_done = Event()
    producer_error = []
    consumer_error = []
    matched = []
    push_results = []

    def fetch_detail(job):
        if stop_event.is_set():
            raise _PipelineCancelled
        city_name = str(job.get("cityName") or "").strip().rstrip("市")
        if city_name and city_name in excluded:
            return
        key = browser.job_key(job)
        if key in jobs_by_key:
            return
        jobs_by_key[key] = job
        try:
            card = browser.get_job_card(job)
        except (VerificationRequired, AccessRestricted) as exc:
            exc.partial_cards = list(cards)
            raise
        if card is not None:
            cards.append(card)
            _queue_put(detail_queue, card, stop_event, consumer_done)

    def consume_details():
        batch = []
        try:
            while True:
                item = detail_queue.get()
                try:
                    if item is _PIPELINE_SENTINEL:
                        break
                    batch.append(item)
                    if len(batch) >= MATCH_BATCH_SIZE:
                        matched.extend(
                            match_card_batch(
                                batch,
                                state.user_input,
                                p.exclude_keywords,
                                search_params=p,
                            )
                        )
                        batch.clear()
                finally:
                    detail_queue.task_done()
            if batch and not stop_event.is_set():
                matched.extend(
                    match_card_batch(
                        batch,
                        state.user_input,
                        p.exclude_keywords,
                        search_params=p,
                    )
                )
        except Exception as exc:
            consumer_error.append(exc)
            stop_event.set()
        finally:
            consumer_done.set()

    def produce_details():
        try:
            locations = p.location or (["全国"] if p.location_unlimited else [""])
            for city in locations:
                if stop_event.is_set():
                    raise _PipelineCancelled
                with browser_io_lock:
                    jobs = browser.get_job_list(
                        p.zhi_wei or "", city=city,
                        salary_code=code_book.code_of("salary", p.money or ""),
                        experience_code=code_book.codes_of("experience", p.experience),
                        scale_code=code_book.codes_of("scale", p.scale),
                        degree_code=code_book.codes_of("degree", p.degree),
                        job_type_code=code_book.code_of("job_type", p.job_type or ""),
                        stage_code=code_book.codes_of("stage", p.stage),
                        on_job=fetch_detail,
                    )
                for job in jobs:
                    city_name = str(job.get("cityName") or "").strip().rstrip("市")
                    if city_name and city_name in excluded:
                        continue
                    jobs_by_key.setdefault(browser.job_key(job), job)
        except _PipelineCancelled:
            if not consumer_error:
                producer_error.append(_PipelineCancelled())
        except Exception as exc:
            producer_error.append(exc)
            stop_event.set()
        finally:
            if not consumer_done.is_set():
                try:
                    _queue_put(detail_queue, _PIPELINE_SENTINEL, stop_event, consumer_done, force=True)
                except _PipelineCancelled:
                    pass

    consumer = Thread(target=consume_details, name=f"job-match-{state.task_id[:8]}", daemon=True)
    producer = Thread(target=produce_details, name=f"job-fetch-{state.task_id[:8]}", daemon=True)
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()

    if producer_error and not consumer_error:
        exc = producer_error[0]
        if isinstance(exc, (VerificationRequired, AccessRestricted)):
            partial_cards = getattr(exc, "partial_cards", None)
            cards = [card for card in (partial_cards or cards) if card is not None]
            logger.warning("职位搜索与详情获取中断: task_id=%s saved=%s reason=%s", state.task_id, len(cards), exc)
            return {"jobs": list(jobs_by_key.values()), "job_cards": cards, "matched_jobs": matched,
                    "push_results": [], "result": f"职位读取已停止，保留 {len(cards)} 个职位详情",
                    "error": f"职位详情获取失败: {type(exc).__name__}: {exc}", "status": "cards_failed", "working": False}
        if not isinstance(exc, _PipelineCancelled):
            logger.error("职位搜索与详情获取失败: task_id=%s error=%s", state.task_id, f"{type(exc).__name__}: {exc}")
            return {"jobs": list(jobs_by_key.values()), "job_cards": cards, "matched_jobs": matched,
                    "push_results": [], "result": f"职位读取已停止，保留 {len(cards)} 个职位详情",
                    "error": f"职位搜索或职位详情获取失败: {type(exc).__name__}: {exc}", "status": "search_failed", "working": False}

    if consumer_error:
        exc = consumer_error[0]
        logger.error("职位匹配失败: task_id=%s error=%s", state.task_id, f"{type(exc).__name__}: {exc}")
        return {"jobs": list(jobs_by_key.values()), "job_cards": cards, "matched_jobs": matched,
                "push_results": [], "result": f"职位处理已停止，保留 {len(cards)} 个职位详情",
                "error": f"向量匹配失败: {type(exc).__name__}: {exc}", "status": "match_failed", "working": False}

    matched.sort(key=lambda item: item.get("score", 0), reverse=True)
    try:
        push_results = push_matches(browser, matched, browser_io_lock, state.task_id) if matched else []
    except (VerificationRequired, AccessRestricted) as exc:
        push_results = getattr(exc, "partial_results", push_results)
        return {"jobs": list(jobs_by_key.values()), "job_cards": cards, "matched_jobs": matched,
                "push_results": push_results, "result": f"职位处理已停止，已匹配 {len(matched)} 个，已投递 {len(push_results)} 个",
                "error": f"投递失败: {type(exc).__name__}: {exc}", "status": "push_failed", "working": False}
    except Exception as exc:
        push_results = getattr(exc, "partial_results", push_results)
        logger.exception("职位投递失败: task_id=%s", state.task_id)
        return {"jobs": list(jobs_by_key.values()), "job_cards": cards, "matched_jobs": matched,
                "push_results": push_results, "result": f"职位处理已停止，已匹配 {len(matched)} 个，已投递 {len(push_results)} 个",
                "error": f"投递失败: {type(exc).__name__}: {exc}", "status": "push_failed", "working": False}

    ok = sum(1 for result in push_results if result.get("success"))
    jobs = list(jobs_by_key.values())
    logger.info("职位管线完成: task_id=%s jobs=%s cards=%s matched=%s pushed=%s", state.task_id, len(jobs), len(cards), len(matched), ok)
    return {"jobs": jobs, "job_cards": cards, "matched_jobs": matched, "push_results": push_results,
            "result": f"找到 {len(jobs)} 个职位，获取 {len(cards)} 个详情，匹配 {len(matched)} 个，投递成功 {ok} 个",
            "status": "completed", "working": False, "pipeline_processed": True}
