"""通过正常职位卡片点击读取职位详情的手动烟测。

这个脚本只走网页上的正常流程：打开搜索页，点击职位卡片，然后从详情
区域的 DOM 读取职位描述。它不会直接访问 ``job/card.json``，也不会记录
Cookie、Authorization 或其他登录凭证。

运行示例：

    python test/test_job_detail_click.py --query AI --city 101210100

脚本需要复用已经登录的 Chromium 用户目录。遇到登录页、安全验证页或
访问限制时会立即停止，请人工处理后重新运行。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import socket
import sys
from time import monotonic, sleep
from urllib.parse import urlencode, urlparse

from DrissionPage import ChromiumOptions, ChromiumPage


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
USER_DATA_DIR = BASE_DIR.parent / "user_data" / "user_data"
DEFAULT_RESULT_FILE = BASE_DIR / "jobs_data" / "test_job_detail_click_results.json"
SEARCH_URL = "https://www.zhipin.com/web/geek/jobs"

# 这些关键词用于识别需要停止的安全状态，不用于绕过验证。
RESTRICTED_TEXTS = (
    "访问受限",
    "您的IP存在异常行为",
    "您的 IP 存在异常行为",
    "请勿频繁提交刷新请求",
    "您的账户存在异常行为",
)
VERIFY_PATHS = ("/web/passport/zp/verify.html", "/403.html")

# BOSS 页面版本变化时，详情容器的 class 可能变化，因此按常见容器逐级尝试。
JOB_CARD_SELECTORS = (
    "css:li.job-card-wrapper",
    "css:div.job-card-wrapper",
    "css:.job-card-wrapper",
    "css:.job-card-wrap",
)
DETAIL_SELECTORS = (
    ".job-detail",
    ".job-detail-box",
    ".job-detail-content",
    ".job-sec-text",
    ".job-description",
    '[class*="job-detail"]',
)
DETAIL_MARKERS = (
    "岗位职责",
    "工作职责",
    "任职要求",
    "职位描述",
    "工作内容",
    "岗位要求",
)
AUTH_COOKIE_NAMES = frozenset({"bst", "wt2", "zp_at"})
LOGIN_ENTRY_TEXTS = ("登录/注册", "立即登录", "登录账号，查看更多好职位")


class TestBlocked(RuntimeError):
    """测试因登录、验证或访问限制而无法继续。"""


def _page_text(page: ChromiumPage) -> str:
    """读取页面可见文字，读取失败时返回空字符串。"""
    try:
        value = page.run_js("return document.body ? document.body.innerText : '';" )
    except Exception:
        return ""
    return str(value or "")


def security_state(page: ChromiumPage) -> str | None:
    """返回当前安全状态；正常页面返回 None。"""
    url = page.url or ""
    if any(path in url for path in VERIFY_PATHS):
        return f"浏览器进入安全验证或 403 页面: {urlparse(url).path}"

    text = _page_text(page)
    for marker in RESTRICTED_TEXTS:
        if marker in text:
            return f"页面提示访问受限: {marker}"

    has_captcha = page.run_js(
        """
        const selectors = [
            '.geetest_panel', '.geetest_box', '.geetest_popup',
            '.nc-container', 'iframe[src*="captcha"]',
            'iframe[src*="geetest"]'
        ];
        const visible = (element) => {
            if (!element || !element.isConnected) return false;
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.display !== 'none'
                && style.visibility !== 'hidden'
                && Number(style.opacity) !== 0
                && rect.width > 0 && rect.height > 0;
        };
        return selectors.some((selector) =>
            Array.from(document.querySelectorAll(selector)).some(visible)
        );
        """
    )
    if has_captcha:
        return "页面出现验证码组件"
    return None


def login_diagnostics(page: ChromiumPage) -> dict:
    """收集登录判定信号，只返回 Cookie 名称，不读取 Cookie 值。"""
    page_flag = None
    try:
        page_flag = page.run_js(
            """
            const value = window._PAGE && window._PAGE.isLogin;
            return value === true ? true : value === false ? false : null;
            """
        )
    except Exception:
        pass

    try:
        cookie_names = sorted(
            str(cookie.get("name"))
            for cookie in (page.cookies() or [])
            if cookie.get("name")
        )
    except Exception:
        cookie_names = []

    text = _page_text(page)
    login_entry_hits = [marker for marker in LOGIN_ENTRY_TEXTS if marker in text]
    auth_cookie_names = sorted(set(cookie_names) & AUTH_COOKIE_NAMES)
    # BOSS 页面可能在异步初始化前暂时把 _PAGE.isLogin 设成 false；
    # 已存在会话 Cookie 时，以 Cookie 为准继续验证真实页面能力。
    confirmed = page_flag is True or bool(auth_cookie_names)
    return {
        "confirmed": confirmed,
        "page_is_login": page_flag,
        "auth_cookie_names": auth_cookie_names,
        "login_entry_hits": login_entry_hits,
    }


def visible_job_cards(page: ChromiumPage):
    """按常见 DOM 选择器查找职位卡片。"""
    for selector in JOB_CARD_SELECTORS:
        cards = page.eles(selector)
        if cards:
            return cards, selector
    return [], ""


def extract_detail_dom(page: ChromiumPage) -> dict:
    """从当前页面可见详情区域提取标题和正文，不调用职位详情接口。"""
    result = page.run_js(
        """
        const selectors = JSON.parse(arguments[0]);
        const markers = JSON.parse(arguments[1]);
        const visible = (element) => {
            if (!element || !element.isConnected) return false;
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.display !== 'none'
                && style.visibility !== 'hidden'
                && Number(style.opacity) !== 0
                && rect.width > 0 && rect.height > 0;
        };
        const candidates = [];
        for (const selector of selectors) {
            for (const element of document.querySelectorAll(selector)) {
                if (!visible(element)) continue;
                const text = (element.innerText || '').replace(/\\s+/g, ' ').trim();
                if (text.length < 40) continue;
                const markerHits = markers.filter((marker) => text.includes(marker)).length;
                candidates.push({selector, text, markerHits});
            }
        }
        candidates.sort((left, right) => {
            if (right.markerHits !== left.markerHits) {
                return right.markerHits - left.markerHits;
            }
            return right.text.length - left.text.length;
        });

        const titleElement = document.querySelector('.job-name, .job-title, h1');
        return {
            title: titleElement ? (titleElement.innerText || '').trim() : '',
            detail: candidates[0] || null,
            url: window.location.href,
        };
        """,
        json.dumps(list(DETAIL_SELECTORS), ensure_ascii=False),
        json.dumps(list(DETAIL_MARKERS), ensure_ascii=False),
    )
    return result if isinstance(result, dict) else {"title": "", "detail": None, "url": page.url}


def wait_for_detail(
    page: ChromiumPage,
    timeout: float,
    *,
    expected_title: str = "",
    previous_text: str = "",
) -> dict:
    """等待点击后的详情 DOM 出现，并在期间持续检查风控状态。"""
    deadline = monotonic() + timeout
    expected_title = " ".join(expected_title.split())
    expected_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", expected_title)[:12]
    while monotonic() < deadline:
        blocked = security_state(page)
        if blocked:
            raise TestBlocked(blocked)

        result = extract_detail_dom(page)
        detail = result.get("detail")
        if isinstance(detail, dict) and detail.get("text"):
            detail_text = " ".join(str(detail["text"]).split())
            detail_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", detail_text)
            title_matches = not expected_key or expected_key in detail_key
            changed = not previous_text or detail_text != previous_text
            previous_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", previous_text)
            previous_title_matches = bool(expected_key and expected_key in previous_key)
            if title_matches and (changed or previous_title_matches):
                return result
        sleep(0.25)
    raise TimeoutError(f"点击后 {timeout:.1f}s 内没有发现职位详情 DOM。")


def response_payload(packet) -> dict | None:
    """读取详情请求的 JSON 状态，不返回或保存正文。"""
    if not packet:
        return None
    try:
        response = getattr(packet, "response", None)
        body = getattr(response, "body", None)
    except Exception:
        return None
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None
    return body if isinstance(body, dict) else None


def packet_summary(packet) -> dict | None:
    """只输出自然点击触发的请求摘要，避免把认证信息写入日志。"""
    if not packet:
        return None
    request = getattr(packet, "request", None)
    response = getattr(packet, "response", None)
    url = getattr(request, "url", None) or getattr(response, "url", None) or ""
    status = getattr(response, "status", None)
    payload = response_payload(packet)
    return {
        "path": urlparse(str(url)).path,
        "method": getattr(request, "method", None),
        "status": status,
        "api_code": payload.get("code") if payload else None,
        "api_message": payload.get("message") if payload else None,
    }


def save_results(path: pathlib.Path, result: dict) -> None:
    """增量保存结果，方便在风控或浏览器中断后查看已完成职位。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def free_local_port() -> int:
    """申请一个当前可用的本机端口，避免和已有 Chrome 调试端口冲突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_cards(page: ChromiumPage, timeout: float) -> tuple[list, str]:
    """等待搜索列表卡片出现。"""
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        blocked = security_state(page)
        if blocked:
            raise TestBlocked(blocked)
        cards, selector = visible_job_cards(page)
        if cards:
            return cards, selector
        sleep(0.25)
    raise TimeoutError(f"{timeout:.1f}s 内没有找到职位卡片。")


def restore_job_list(page: ChromiumPage, search_url: str, timeout: float) -> tuple[list, str]:
    """点击详情后回到列表，兼容侧栏详情和独立详情页两种表现。"""
    current_url = page.url or ""
    if current_url != search_url:
        page.back()
        page.wait.load_start()

    # 侧栏详情通常不会改变 URL；按正常用户操作发送 Escape 关闭侧栏。
    try:
        page.actions.key_down("ESC")
        page.actions.key_up("ESC")
    except Exception:
        pass
    return wait_for_cards(page, timeout)


def load_more_cards(page: ChromiumPage, old_count: int, timeout: float) -> tuple[list, str] | None:
    """滚动列表加载下一批卡片；没有新增卡片时返回 None。"""
    try:
        page.actions.scroll(1000000, 0)
    except Exception:
        return None

    deadline = monotonic() + timeout
    while monotonic() < deadline:
        blocked = security_state(page)
        if blocked:
            raise TestBlocked(blocked)
        cards, selector = visible_job_cards(page)
        if len(cards) > old_count:
            return cards, selector
        sleep(0.25)
    return None


def run_test(
    query: str,
    city: str,
    timeout: float,
    max_jobs: int,
    result_path: pathlib.Path,
) -> dict:
    """逐个点击职位卡片并返回所有已完成结果。"""
    options = ChromiumOptions()
    # 测试默认必须让用户看到真实浏览器和安全验证状态。
    options.headless(False)
    options.set_local_port(free_local_port())
    options.set_user_data_path(str(USER_DATA_DIR))
    page = ChromiumPage(addr_or_opts=options)
    listener_started = False
    keep_page_open = False

    try:
        url = SEARCH_URL + "?" + urlencode({"query": query, "city": city})
        page.get(url)
        page.wait.load_start()

        blocked = security_state(page)
        if blocked:
            raise TestBlocked(blocked)
        login = login_diagnostics(page)
        if not login["confirmed"]:
            raise TestBlocked(
                "无法确认 BOSS 登录状态；"
                f"诊断={json.dumps(login, ensure_ascii=False, separators=(',', ':'))}"
            )
        print(
            "登录状态诊断: "
            f"page_is_login={login['page_is_login']} "
            f"auth_cookie_names={','.join(login['auth_cookie_names']) or '-'} "
            f"login_entry_hits={','.join(login['login_entry_hits']) or '-'}"
        )

        cards, card_selector = wait_for_cards(page, timeout)
        results = []
        index = 0
        stopped_reason = None

        while True:
            if max_jobs and index >= max_jobs:
                break
            blocked = security_state(page)
            if blocked:
                stopped_reason = blocked
                break

            cards, card_selector = visible_job_cards(page)
            if index >= len(cards):
                loaded = load_more_cards(page, len(cards), min(timeout, 5))
                if loaded is None:
                    break
                cards, card_selector = loaded
                if index >= len(cards):
                    break

            card = cards[index]
            # DrissionPage 4.x 的元素文本通过 ``text`` 属性读取。
            card_text = (card.text or "").strip()
            expected_title = card_text.splitlines()[0].strip() if card_text else ""
            previous_detail = extract_detail_dom(page).get("detail") or {}
            previous_detail_text = " ".join(str(previous_detail.get("text", "")).split())
            page.listen.start(targets=["/wapi/zpgeek/job/card.json"], is_regex=False)
            listener_started = True
            card.click()

            # 先检查自然点击产生的响应，code=36 时不再等待或点击下一个职位。
            try:
                packet = page.listen.wait(timeout=min(timeout, 5))
            except Exception:
                # 有些页面版本直接复用列表数据或走其他详情请求；只要 DOM
                # 正常出现，网络包观测缺失不应阻断点击测试。
                packet = None
            summary = packet_summary(packet)
            if summary and summary.get("api_code") == 36:
                stopped_reason = f"详情请求触发 code=36: {summary.get('api_message') or '账户存在异常行为'}"
                break

            detail = wait_for_detail(
                page,
                timeout,
                expected_title=expected_title,
                previous_text=previous_detail_text,
            )
            detail_text = ((detail.get("detail") or {}).get("text") or "").strip()
            if len(detail_text) < 40:
                raise AssertionError("详情 DOM 文本过短，无法确认读取到职位正文。")

            results.append({
                "card_index": index,
                "card_selector": card_selector,
                "card_text_preview": card_text[:200],
                # 详情页可能保留列表侧栏的旧 h1；卡片首行才对应本次点击的职位。
                "detail_title": card_text.splitlines()[0].strip() if card_text else detail.get("title", ""),
                "detail_selector": (detail.get("detail") or {}).get("selector", ""),
                "detail_chars": len(detail_text),
                "detail_preview": detail_text[:300],
                "detail_url": detail.get("url", page.url),
                "observed_request": summary,
            })
            save_results(result_path, {
                "status": "running",
                "search_url": url,
                "completed": len(results),
                "results": results,
            })
            print(f"已完成 {index + 1} 个职位: {detail.get('title') or card_text[:40]}")
            index += 1

            if listener_started:
                page.listen.stop()
                listener_started = False
            cards, card_selector = restore_job_list(page, url, timeout)

        final = {
            "status": "blocked" if stopped_reason else "passed",
            "stopped_reason": stopped_reason,
            "search_url": url,
            "completed": len(results),
            "results": results,
        }
        save_results(result_path, final)
        keep_page_open = bool(stopped_reason)
        return final
    except TestBlocked:
        # 让用户能直接看到登录/安全验证页面并人工处理。
        keep_page_open = True
        raise
    finally:
        if listener_started:
            try:
                page.listen.stop()
            except Exception:
                pass
        if not keep_page_open:
            page.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试通过职位卡片点击读取详情 DOM")
    parser.add_argument("--query", default="AI", help="职位搜索关键词")
    parser.add_argument("--city", default="101210100", help="BOSS 城市编码")
    parser.add_argument("--timeout", type=float, default=15, help="等待详情 DOM 的秒数")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="最多点击数量，0 表示持续点击当前搜索结果中的全部职位",
    )
    parser.add_argument(
        "--result-file",
        type=pathlib.Path,
        default=DEFAULT_RESULT_FILE,
        help=f"增量结果文件，默认 {DEFAULT_RESULT_FILE}",
    )
    parser.add_argument(
        "--keep-open-on-block",
        action="store_true",
        help="遇到登录或安全验证时保持浏览器和脚本运行，按回车退出",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_test(
            args.query,
            args.city,
            args.timeout,
            args.max_jobs,
            args.result_file,
        )
    except TestBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        if args.keep_open_on_block:
            print("浏览器保持打开；完成人工登录/验证后，在此终端按回车退出。", file=sys.stderr)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        return 2
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
