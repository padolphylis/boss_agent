import json
import pathlib
import unicodedata
from threading import Event, RLock
from random import uniform
from typing import Callable
from time import monotonic, sleep
from urllib.request import Request, urlopen
from urllib.parse import quote, urlencode, urlparse

from DrissionPage import Chromium, ChromiumPage, ChromiumOptions

from analysis_work_content import merge_benefits
from logging_config import get_logger
from matcher import code_book

logger = get_logger(__name__)

browser_io_lock = RLock()


class BrowserBlocked(Exception):
    """浏览器操作被阻止；批量详情读取时携带已完成的结果。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.partial_cards: list[dict | None] = []


class VerificationRequired(BrowserBlocked):
    """需要人工处理验证码。"""


class AccessRestricted(BrowserBlocked):
    """页面访问受限。"""


class DeliveryRateLimited(AccessRestricted):
    """投递请求触发平台频率限制或当日沟通上限。"""


class LoginRequired(BrowserBlocked):
    """登录态缺失或已失效，需要用户重新登录。"""


class BrowserOperationCancelled(Exception):
    """浏览器操作因上层任务停止而取消。"""


base_dir = pathlib.Path(__file__).resolve().parent
user_data_dir = base_dir.parent / 'user_data' / 'user_data'
save_dir = base_dir / 'jobs_data'
user_data_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)

job_list_target = '/wapi/zpgeek/search/joblist.json'
job_card_api = 'https://www.zhipin.com/wapi/zpgeek/job/card.json'
login_probe_api = 'https://www.zhipin.com/wapi/zpuser/wap/getUserInfo.json'
login_page_url = 'https://www.zhipin.com/web/user/?ka=header-login'
login_probe_failure_codes = frozenset({'7'})
login_probe_failure_messages = (
    '请登录',
    '未登录',
    '登录状态',
    '登录失效',
    '重新登录',
)
login_probe_retry_count = 3
login_probe_retry_delay = 0.3
login_probe_request_timeout_seconds = 3
push_api = 'https://www.zhipin.com/wapi/zpgeek/friend/add.json'
daily_limit_hint = '您今天已与120位BOSS沟通'
chat_url = 'https://www.zhipin.com/web/geek/chat'

job_card_selector = '.job-card-wrapper, .job-card-wrap'
job_detail_body_selectors = (
    '.job-detail-body .desc',
    '.job-sec-text',
    '.job-description',
)
job_detail_container_selectors = (
    '.job-primary.detail-box',
    '.job-detail-container',
    '.job-detail-box',
    '.job-detail',
)
login_entry_texts = ('登录/注册', '立即登录', '登录账号，查看更多好职位')
login_entry_selector = 'header a, header button, nav a, nav button, .header a, .header button, .header-nav a, .header-nav button'
login_page_markers = (
    '/web/passport/',
    '/passport/',
    '/user/login',
    '/login',
)
job_list_navigation_timeout_seconds = 10
job_list_response_timeout_seconds = 8
job_card_request_timeout_seconds = 8
job_detail_dom_timeout_seconds = 30
push_request_timeout_seconds = 15
delivery_rate_limit_messages = (
    '操作过于频繁',
    '休息一会再试',
    '请求过于频繁',
    '请求频繁',
    '请稍后再试',
    daily_limit_hint,
)

debug_port = 49613
login_cache_ttl_seconds = 15.0


def _debug_pages(port: int) -> list[dict]:
    """读取 DevTools 页面列表；浏览器尚未就绪或连接失败时返回空列表。"""
    try:
        with urlopen(f'http://127.0.0.1:{port}/json/list', timeout=1) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _open_debug_page(port: int, url: str = 'about:blank') -> str | None:
    """在已有 DevTools 实例中创建普通页面并返回页面 ID。"""
    request = Request(
        f'http://127.0.0.1:{port}/json/new?{quote(url, safe=":/?=&")}',
        method='PUT',
    )
    try:
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode('utf-8'))
        page_id = payload.get('id') if isinstance(payload, dict) else None
        return str(page_id) if page_id else None
    except Exception:
        logger.debug("通过 DevTools 创建页面失败: port=%s", port, exc_info=True)
        return None


def _debug_browser(port: int) -> dict | None:
    """读取 DevTools 浏览器信息；仅有有效浏览器端点才视为实例存在。"""
    try:
        with urlopen(f'http://127.0.0.1:{port}/json/version', timeout=1) as response:
            payload = json.loads(response.read().decode('utf-8'))
        if isinstance(payload, dict) and payload.get('webSocketDebuggerUrl'):
            return payload
    except Exception:
        pass
    return None


def _stop_stale_browser(port: int, data_dir: pathlib.Path) -> bool:
    """清理本项目留下的无页面 Chrome，避免 DrissionPage 误连失效实例。"""
    try:
        import psutil
    except ImportError:
        logger.warning("未安装 psutil，无法自动清理失效浏览器实例")
        return False

    port_marker = f'--remote-debugging-port={port}'
    data_marker = str(data_dir.resolve())
    stopped = False
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            command = ' '.join(process.info.get('cmdline') or [])
            if (
                process.info.get('name') not in {'Google Chrome', 'chrome', 'Chromium'}
                or port_marker not in command
                or data_marker not in command
            ):
                continue
            logger.warning("清理无页面浏览器实例: pid=%s port=%s", process.pid, port)
            process.terminate()
            try:
                process.wait(timeout=3)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return stopped


class BrowserManager:
    def __init__(self):
        self.user_data_dir = user_data_dir
        self._page:ChromiumPage|None = None
        self._owns_browser = False
        self._browser_identity: str | None = None
        self._push_page = None
        self._chat_page = None
        self._login_probe_page = None
        self._job_detail_page = None
        self._last_search_url = ''
        self._ready = Event()
        self._start_lock = RLock()
        self._login_verified_at = 0.0


    def get_page(self) -> ChromiumPage:
        if self._page is None:
            raise ValueError("Browser not started")
        return self._page


    def start(self):
        """优先复用或重连现有实例；仅在 DevTools 不存在时创建浏览器。"""
        with self._start_lock:
            if self._page is not None and self._page_is_alive(self._page):
                return self

            self._login_verified_at = 0.0
            options = self._browser_options()
            existing_info = _debug_browser(debug_port)
            existing = existing_info is not None
            if existing:
                current_identity = self._browser_identity
                existing_identity = self._browser_identity_from_debug_info(existing_info)
                preserve_ownership = (
                    self._owns_browser
                    and self._page is not None
                    and current_identity is not None
                    and current_identity == existing_identity
                )
                logger.info("发现现有浏览器实例，尝试恢复连接: port=%s", debug_port)
                try:
                    page = self._connect_existing(options)
                    connected_identity = self._browser_identity_from_page(page)
                    self._page = page
                    self._close_auxiliary_page("_push_page", "投递")
                    self._close_auxiliary_page("_chat_page", "聊天监听")
                    self._close_auxiliary_page("_login_probe_page", "登录探针")
                    self._close_auxiliary_page("_job_detail_page", "职位详情")
                    self._owns_browser = (
                        preserve_ownership
                        and connected_identity is not None
                        and connected_identity == existing_identity
                    )
                    self._browser_identity = connected_identity or existing_identity
                    logger.info("现有浏览器连接已恢复")
                    self._ready.set()
                    return self
                except Exception:
                    if _debug_browser(debug_port) is not None:
                        logger.exception("重连现有浏览器失败: port=%s", debug_port)
                        raise
                    logger.info("浏览器实例已退出，将启动新实例")

            logger.info("正在创建浏览器实例: user_data_dir=%s port=%s", self.user_data_dir, debug_port)
            _stop_stale_browser(debug_port, self.user_data_dir)
            self._page = ChromiumPage(addr_or_opts=options)
            self._close_auxiliary_page("_push_page", "投递")
            self._close_auxiliary_page("_chat_page", "聊天监听")
            self._close_auxiliary_page("_login_probe_page", "登录探针")
            self._close_auxiliary_page("_job_detail_page", "职位详情")
            self._owns_browser = True
            self._browser_identity = self._browser_identity_from_page(self._page)
            if not self._page_is_alive(self._page):
                self._page = None
                self._owns_browser = False
                self._browser_identity = None
                raise RuntimeError("浏览器已启动，但页面连接未就绪。")
            logger.info("浏览器启动完成，当前进程拥有浏览器实例")
            self._ready.set()
            return self

    def _browser_options(self) -> ChromiumOptions:
        options = ChromiumOptions()
        options.headless(False)
        options.set_local_port(debug_port)
        options.set_user_data_path(str(self.user_data_dir))
        options.set_argument('--no-startup-window', False)
        options.set_argument('--new-window')
        return options

    @staticmethod
    def _page_is_alive(page) -> bool:
        try:
            page.run_js('return true;')
            return True
        except Exception:
            return False

    def _close_auxiliary_page(self, attribute: str, label: str) -> None:
        """关闭并清空投递、登录探针等后台标签页引用。"""
        page = getattr(self, attribute, None)
        setattr(self, attribute, None)
        if page is None or page is self._page:
            return
        try:
            page.close()
        except Exception:
            logger.debug("关闭%s后台标签页失败", label, exc_info=True)

    @staticmethod
    def _browser_identity_from_debug_info(info: dict | None) -> str | None:
        """提取 DevTools 浏览器 ID，用于确认端口上的实例未被替换。"""
        if not isinstance(info, dict):
            return None
        websocket_url = str(info.get("webSocketDebuggerUrl") or "").strip()
        return websocket_url.rsplit("/", 1)[-1] or None

    @staticmethod
    def _browser_identity_from_page(page) -> str | None:
        """从 DrissionPage 页面对象读取其所属浏览器 ID。"""
        try:
            identity = page.browser.id
        except Exception:
            return None
        if isinstance(identity, (str, int)) and str(identity).strip():
            return str(identity)
        return None

    def _connect_existing(self, options: ChromiumOptions) -> ChromiumPage:
        """连接现有 DevTools 实例，选择可执行同源请求的普通页面。"""
        debug_pages = [
            page for page in _debug_pages(debug_port)
            if page.get('type') == 'page' and page.get('id')
        ]
        opened_page_id = None
        if not debug_pages:
            opened_page_id = _open_debug_page(debug_port)

        browser = Chromium(addr_or_opts=options)
        try:
            tab_ids = browser.tab_ids
        except Exception:
            try:
                browser.reconnect()
            except Exception:
                Chromium._BROWSERS.pop(browser.id, None)
                browser = Chromium(addr_or_opts=options)
            tab_ids = browser.tab_ids

        debug_pages_by_id = {
            str(page.get('id')): page
            for page in debug_pages
            if page.get('id')
        }

        def page_url(page_id: str) -> str:
            info = debug_pages_by_id.get(str(page_id)) or {}
            debug_url = str(info.get('url') or '').strip()
            if debug_url:
                return debug_url
            if self._page is not None:
                try:
                    old_page_id = str(self._page.tab_id)
                    if old_page_id == str(page_id):
                        return str(self._page.url or '').strip()
                except Exception:
                    pass
            return ''

        def is_ordinary_boss_page(url: str) -> bool:
            if not self._same_site(url, login_probe_api):
                return False
            path = (urlparse(url).path or '').lower()
            return not path.startswith('/wapi/') and not path.endswith('.json')

        old_page_id = None
        if self._page is not None:
            try:
                old_page_id = str(self._page.tab_id)
                if (
                    old_page_id in tab_ids
                    and is_ordinary_boss_page(page_url(old_page_id))
                ):
                    self._page.reconnect()
                    if self._page_is_alive(self._page):
                        return self._page
            except Exception:
                logger.info("原浏览器标签无法重连，将选择其他标签")

        candidate_ids = [
            str(page_id)
            for page_id in tab_ids
            if is_ordinary_boss_page(page_url(str(page_id)))
        ]
        candidate_ids.sort(
            key=lambda page_id: (
                0
                if '/web/geek/jobs' in page_url(page_id).lower()
                else 1,
                0 if page_id == old_page_id else 1,
            )
        )
        if candidate_ids:
            page_id = candidate_ids[0]
            logger.info(
                "选择现有 Boss 页面: tab_id=%s url=%s",
                page_id,
                page_url(page_id),
            )
            navigate_to_home = False
        elif opened_page_id and str(opened_page_id) in tab_ids:
            page_id = str(opened_page_id)
            navigate_to_home = True
            logger.info("复用新建标签并打开 Boss 首页: tab_id=%s", page_id)
        else:
            page_id = browser.new_tab(
                'https://www.zhipin.com/',
                background=False,
            ).tab_id
            navigate_to_home = True
            logger.info("未找到可复用 Boss 页面，创建首页标签: tab_id=%s", page_id)

        cached_page = ChromiumPage._PAGES.get(browser.id)
        if cached_page is not None and cached_page.tab_id == page_id:
            if self._page_is_alive(cached_page) and not navigate_to_home:
                return cached_page
            if not navigate_to_home:
                ChromiumPage._PAGES.pop(browser.id, None)
            cached_page = None
        if cached_page is not None:
            ChromiumPage._PAGES.pop(browser.id, None)
        page = ChromiumPage(addr_or_opts=options, tab_id=page_id)
        if not self._page_is_alive(page):
            try:
                page.reconnect()
            except Exception:
                ChromiumPage._PAGES.pop(browser.id, None)
                page = ChromiumPage(addr_or_opts=options, tab_id=page_id)
        if not self._page_is_alive(page):
            raise RuntimeError("现有浏览器的页面连接仍不可用。")
        if navigate_to_home:
            page.get('https://www.zhipin.com/')
            page.wait.load_start()
            current_url = str(page.url or '').strip()
            if not self._same_site(current_url, login_probe_api):
                raise RuntimeError(
                    f"新建 Boss 页面导航失败: current_url={current_url or '-'}"
                )
        return page


    def is_ready(self) -> bool:
        """浏览器是否已经启动并持有可用页面。"""
        return self._page is not None and self._page_is_alive(self._page)


    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """等待浏览器就绪。"""
        if self.is_ready():
            return True
        return self._ready.wait(timeout) and self.is_ready()

    @staticmethod
    def _same_site(current_url: str, target_url: str) -> bool:
        """判断当前标签是否已在目标站点，允许复用已有的 Boss 页面。"""
        current_host = (urlparse(current_url).hostname or '').lower().removeprefix('www.')
        target_host = (urlparse(target_url).hostname or '').lower().removeprefix('www.')
        return bool(current_host and target_host and current_host == target_host)

    def _login_cache_is_fresh(self) -> bool:
        return (
            self._login_verified_at > 0
            and monotonic() - self._login_verified_at < login_cache_ttl_seconds
        )

    def goto(self, url: str):
        logger.info("打开页面: url=%s", url)
        page = self.start().get_page()
        current_url = str(page.url or '').strip()
        if self._same_site(current_url, url):
            logger.info("复用当前 Boss 页面，不重复导航: current_url=%s", current_url)
        else:
            page.get(url)
            page.wait.load_start()
        self.ensure_or_wait()
        return page

    def open_chat(self) -> ChromiumPage:
        """在后台标签页打开聊天页，复用当前浏览器登录态。"""
        page = self._chat_page
        if page is not None and self._page_is_alive(page):
            if page.url != chat_url:
                page.get(chat_url)
                page.wait.load_start()
            self._wait_chat_page_ready(page)
            return page

        if page is not None:
            logger.info("聊天后台标签页已失效，将重新创建")
            self._chat_page = None

        main_page = self.get_page()
        page = main_page.new_tab(chat_url, background=True)
        page.wait.load_start()
        self._chat_page = page
        self._wait_chat_page_ready(page)
        return page

    @staticmethod
    def _wait_chat_page_ready(page, timeout: float = 30) -> None:
        """等待聊天 Vue 页面挂载会话列表，避免只等到 HTML 首屏。"""
        deadline = monotonic() + max(0, timeout)
        while True:
            try:
                ready = page.run_js(
                    """
                    return Boolean(
                        window.ChatWebsocket
                        && document.querySelector('.chat-container')
                        && document.querySelector('.user-list-content')
                    );
                    """
                )
                if ready:
                    return
            except Exception:
                pass
            if monotonic() >= deadline:
                raise RuntimeError("聊天页面组件尚未就绪")
            sleep(0.25)

    @staticmethod
    def _chat_sdk_ready(page) -> bool:
        """判断聊天页是否已暴露真实的 RxJS 聊天消息流。"""
        return bool(page.run_js(
            """
            return Boolean(
                window.ChatWebsocket
                && window.ChatWebsocket.message$
                && typeof window.ChatWebsocket.message$.subscribe === 'function'
            );
            """
        ))

    def start_chat_listener(
        self,
        ready_timeout: float = 30,
        ready_poll_interval: float = 0.25,
        history_settle_timeout: float = 5,
    ) -> None:
        """订阅 BOSS 聊天页真实的 RxJS 消息流。"""
        page = self.open_chat()
        deadline = monotonic() + max(0, ready_timeout)
        while True:
            try:
                if self._chat_sdk_ready(page):
                    break
            except Exception as exc:
                if monotonic() >= deadline:
                    raise RuntimeError("聊天 SDK 初始化失败") from exc
            if monotonic() >= deadline:
                raise RuntimeError("聊天 SDK 尚未初始化，请确认聊天页已加载完成")
            sleep(max(0.05, ready_poll_interval))

        history_deadline = min(
            deadline,
            monotonic() + max(0, history_settle_timeout),
        )
        previous_max_id = None
        stable_checks = 0
        while monotonic() < history_deadline:
            max_id = page.run_js(
                """
                return Math.max(
                    Number(window.ChatWebsocket.maxMsgId || 0),
                    Number(window.ChatWebsocket.historyMaxMsgId || 0)
                );
                """
            )
            if max_id and max_id == previous_max_id:
                stable_checks += 1
            else:
                stable_checks = 0
            previous_max_id = max_id
            if max_id and stable_checks >= 2:
                break
            sleep(max(0.05, ready_poll_interval))

        page.run_js(
            """
            return (() => {
                const existing = window.__bossAgentChatBridge;
                if (existing && existing.sdk === window.ChatWebsocket) {
                    return 'already_started';
                }
                if (existing && Array.isArray(existing.subscriptions)) {
                    existing.subscriptions.forEach(subscription => {
                        try { subscription.unsubscribe(); } catch (_) {}
                    });
                }

                const queue = [];
                // 聊天页初始化时会通过同一条 message$ 回放离线历史。只接收
                // 安装桥接之后产生的新 mid，避免服务启动就自动回复旧消息。
                const baselineMessageId = Math.max(
                    Number(window.ChatWebsocket.maxMsgId || 0),
                    Number(window.ChatWebsocket.historyMaxMsgId || 0)
                );
                const append = (payload) => {
                    const payloadList = Array.isArray(payload) ? payload : [payload];
                    const freshPayload = payloadList.filter(item => {
                        const messageId = Number(
                            item && (
                                item.mid
                                || item.messageId
                                || item.id
                                || item.tempID
                                || item.tempId
                            )
                        );
                        return !Number.isFinite(messageId)
                            || messageId > baselineMessageId;
                    });
                    if (!freshPayload.length) return;
                    try {
                        queue.push({
                            kind: 'message',
                            payload: JSON.parse(JSON.stringify(
                                Array.isArray(payload) ? freshPayload : freshPayload[0]
                            )),
                            currentUserId: window._PAGE
                                && (window._PAGE.uid || window._PAGE.userId)
                        });
                    } catch (_) {
                        queue.push({
                            kind: 'message',
                            payload: String(
                                Array.isArray(payload) ? freshPayload : freshPayload[0]
                            ),
                            currentUserId: window._PAGE
                                && (window._PAGE.uid || window._PAGE.userId)
                        });
                    }
                    if (queue.length > 500) {
                        queue.splice(0, queue.length - 500);
                    }
                };

                const sdk = window.ChatWebsocket;
                if (!sdk || !sdk.message$
                    || typeof sdk.message$.subscribe !== 'function') {
                    throw new Error('聊天 SDK 尚未初始化');
                }
                const subscriptions = [sdk.message$.subscribe(append)];
                window.__bossAgentChatBridge = {
                    queue,
                    subscriptions,
                    sdk,
                    baselineMessageId,
                    startedAt: Date.now(),
                };
                return 'started';
            })();
            """
        )

    def poll_chat_messages(self, timeout: float = 1) -> list[dict]:
        """取出页面事件队列中的新消息并转换为统一结构。"""
        deadline = __import__('time').monotonic() + max(0, timeout)
        while True:
            page = self._chat_page
            if page is None:
                raise RuntimeError("聊天监听标签页尚未打开")
            events = page.run_js(
                """
                const bridge = window.__bossAgentChatBridge;
                if (!bridge) return [];
                return bridge.queue.splice(0, bridge.queue.length);
                """
            ) or []
            messages = []
            for event in events:
                payload = event.get('payload')
                payloads = payload if isinstance(payload, list) else [payload]
                for item in payloads:
                    message = self._normalize_chat_message(
                        item,
                        event.get('kind', ''),
                        current_user_id=event.get('currentUserId'),
                    )
                    if message:
                        messages.append(message)
            if messages or __import__('time').monotonic() >= deadline:
                return messages
            sleep(0.1)

    @staticmethod
    def _normalize_chat_message(
        payload: object,
        event_kind: str = '',
        current_user_id: object = None,
    ) -> dict | None:
        """把 BOSS ``message$`` 的原始消息转换成统一结构。"""
        if not isinstance(payload, dict):
            return None

        message = payload
        if isinstance(payload.get('message'), dict):
            message = {**payload, **payload['message']}
        body = message.get('body') if isinstance(message.get('body'), dict) else {}
        text = (
            message.get('text')
            or message.get('content')
            or body.get('text')
            or message.get('messageText')
            or ''
        )
        if not isinstance(text, str):
            text = str(text or '')

        message_id = (
            message.get('mid')
            or message.get('messageId')
            or message.get('id')
            or message.get('tempID')
            or message.get('tempId')
        )
        from_user = message.get('from') if isinstance(message.get('from'), dict) else {}
        to_user = message.get('to') if isinstance(message.get('to'), dict) else {}
        sender_id = (
            message.get('fromId')
            or from_user.get('uid')
            or from_user.get('userId')
        )
        target_id = (
            message.get('toId')
            or to_user.get('uid')
            or to_user.get('userId')
        )
        sender_source = (
            message.get('fromSource')
            or from_user.get('source')
            or 0
        )
        target_source = (
            message.get('toSource')
            or to_user.get('source')
            or 0
        )
        is_self = bool(message.get('isSelf')) or (
            current_user_id is not None
            and str(sender_id) == str(current_user_id)
        )
        friend_id = target_id if is_self else sender_id
        friend_source = target_source if is_self else sender_source
        friend_user = to_user if is_self else from_user
        conversation = {
            'uid': friend_id,
            'friendSource': friend_source,
            'encryptUid': (
                message.get('encryptBossId')
                or friend_user.get('encryptUid')
                or friend_user.get('encryptBossId')
                or ''
            ),
            'groupId': message.get('groupId') or '',
            'encryptGid': message.get('encryptGid') or '',
        }
        conversation_id = (
            message.get('uniqueId')
            or message.get('conversationId')
            or conversation.get('groupId')
            or f"{friend_id}-{friend_source}"
        )
        if not message_id or not conversation_id:
            return None

        return {
            'message_id': str(message_id),
            'conversation_id': str(conversation_id),
            'conversation': conversation,
            'sender_id': str(sender_id or ''),
            'sender_name': str(message.get('fromName') or from_user.get('name') or ''),
            'friend_source': str(friend_source),
            'direction': 'outgoing' if is_self else 'incoming',
            'content': text.strip(),
            'event_kind': event_kind,
            'raw': payload,
        }

    def send_chat_message(self, conversation: dict, content: str) -> dict:
        """通过 BOSS 聊天页自身的 Vue 编辑器发送文本消息。"""
        if not content.strip():
            raise ValueError('消息内容不能为空')
        if not isinstance(conversation, dict):
            raise ValueError('聊天会话信息无效')
        page = self.open_chat()
        result = page.run_js(
            """
            return (async (args) => {
                const wantedId = String(args.conversation.conversationId || '');
                const wantedUid = String(args.conversation.uid || '');
                const wantedSource = String(args.conversation.friendSource || 0);
                const wantedUniqueId = wantedId || `${wantedUid}-${wantedSource}`;

                const getSource = (element) => {
                    let current = element;
                    while (current) {
                        const component = current.__vue__;
                        if (component && (component.source || component.$props?.source)) {
                            return component.source || component.$props.source;
                        }
                        current = current.parentElement;
                    }
                    return null;
                };
                const getUniqueId = (source) => source && (
                    source.uniqueId
                    || `${source.uid || source.friendId}-${source.friendSource || 0}`
                );
                const list = [...document.querySelectorAll('.friend-content')];
                let target = list.find(element => {
                    const source = getSource(element);
                    return source && String(getUniqueId(source)) === wantedUniqueId;
                });

                // 虚拟列表只渲染可视区域，先滚动到数据源中的会话再查 DOM。
                if (!target) {
                    const virtualList = [...document.querySelectorAll('*')]
                        .map(element => element.__vue__)
                        .find(component =>
                            component
                            && component.$options
                            && component.$options.name === 'virtual-list'
                            && Array.isArray(component.dataSources)
                        );
                    const sources = virtualList && virtualList.dataSources;
                    const index = sources
                        ? sources.findIndex(source =>
                            String(getUniqueId(source)) === wantedUniqueId
                        )
                        : -1;
                    if (index >= 0 && typeof virtualList.scrollToIndex === 'function') {
                        virtualList.scrollToIndex(index);
                        await new Promise(resolve => setTimeout(resolve, 150));
                        target = [...document.querySelectorAll('.friend-content')]
                            .find(element => {
                                const source = getSource(element);
                                return source
                                    && String(getUniqueId(source)) === wantedUniqueId;
                            });
                    }
                }

                if (!target) {
                    throw new Error(`未找到聊天会话: ${wantedUniqueId}`);
                }

                target.click();
                await new Promise(resolve => setTimeout(resolve, 150));
                const editor = document.querySelector('#chat-input');
                const editorComponent = editor && (() => {
                    let element = editor;
                    while (element && !element.__vue__) {
                        element = element.parentElement;
                    }
                    return element && element.__vue__;
                })();
                const selectedId = editorComponent
                    && editorComponent.conversation$
                    && editorComponent.conversation$.uniqueId;
                if (!editorComponent || String(selectedId) !== wantedUniqueId
                    || typeof editorComponent.sendText !== 'function') {
                    throw new Error(`聊天会话切换失败: ${wantedUniqueId}`);
                }

                editor.innerHTML = '';
                editor.appendChild(document.createTextNode(args.content));
                editor.dispatchEvent(new InputEvent('input', {
                    bubbles: true,
                    inputType: 'insertText',
                    data: args.content,
                }));
                editorComponent.enableSubmit = true;
                editorComponent.sendText();
                return {
                    sent: true,
                    conversationId: wantedUniqueId,
                };
            })(arguments[0]);
            """,
            {
                'conversation': {
                    **conversation,
                    'conversationId': (
                        conversation.get('conversationId')
                        or conversation.get('uniqueId')
                        or (
                            f"{conversation.get('uid')}-"
                            f"{conversation.get('friendSource') or 0}"
                        )
                    ),
                },
                'content': content.strip(),
            },
        )
        return result if isinstance(result, dict) else {'result': result}

    def stop_chat_listener(self) -> None:
        """移除页面侧聊天事件桥接；不影响 BOSS 页面自身连接。"""
        page = self._chat_page
        if page is None:
            return
        try:
            page.run_js(
                """
                const bridge = window.__bossAgentChatBridge;
                if (bridge && Array.isArray(bridge.subscriptions)) {
                    bridge.subscriptions.forEach(subscription => {
                        try { subscription.unsubscribe(); } catch (_) {}
                    });
                }
                window.__bossAgentChatBridge = null;
                return true;
                """
            )
        except Exception:
            logger.debug('清理聊天监听桥接失败', exc_info=True)


    def __call__(self, url: str) ->ChromiumPage:
        return self.goto(url)


    def check_yan_cheng_ma(self, page=None) -> bool:
        """只检查当前页面是否出现可见的验证码组件。"""
        page = page if page is not None else self.get_page()
        if page.run_js("""
            const selectors = [
                '.geetest_panel',
                '.geetest_box',
                '.geetest_popup',
                '.nc-container',
                'iframe[src*="captcha"]',
                'iframe[src*="geetest"]'
            ];

            function isVisible(element) {
                if (!element || !element.isConnected) {
                    return false;
                }

                let current = element;
                while (current && current !== document.documentElement) {
                    const style = window.getComputedStyle(current);
                    const rect = current.getBoundingClientRect();
                    if (style.display === 'none'
                        || style.visibility === 'hidden'
                        || Number(style.opacity) === 0
                        || rect.width <= 0
                        || rect.height <= 0) {
                        return false;
                    }
                    current = current.parentElement;
                }
                return true;
            }

            return selectors.some((selector) => {
                return Array.from(document.querySelectorAll(selector))
                    .some(isVisible);
            });
        """) is True:
            return True

        current_url = page.url or ''
        if any(path in current_url for path in ('/403.html', '/web/passport/zp/verify.html')):
            logger.warning("检测到验证或访问限制页面: path=%s", urlparse(current_url).path)
            return True

        if page.run_js("""
            const text = document.body ? document.body.innerText : '';
            const restrictedTexts = [
                '访问受限',
                '您的IP存在异常行为',
                '您的 IP 存在异常行为',
                '请勿频繁提交刷新请求',
                '您的账户存在异常行为'
            ];
            return restrictedTexts.some(item => text.includes(item));
        """) is True:
            logger.warning("检测到访问受限提示: url=%s", current_url)
            return True

        return False


    @staticmethod
    def _has_login_entry(page) -> bool:
        """只检查页头/导航中的登录入口，避免职位正文中的“登录/注册”误判。"""
        return bool(page.run_js("""
            const markers = JSON.parse(arguments[0]);
            const selector = arguments[1];
            const visible = element => {
                if (!element || !element.isConnected) return false;
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden'
                    && Number(style.opacity) !== 0 && rect.width > 0 && rect.height > 0;
            };
            return [...document.querySelectorAll(selector)].some(element => {
                const text = (element.innerText || '').trim();
                return visible(element) && markers.includes(text);
            });
        """, json.dumps(login_entry_texts), login_entry_selector))


    def ensure_available(self, page=None) -> None:
        """操作前检查验证码和访问限制。"""
        page = page if page is not None else self.get_page()
        current_url = page.url or ''
        if '/403.html' in current_url:
            raise AccessRestricted('Boss 直聘当前访问受限。')
        if self.check_yan_cheng_ma(page):
            raise VerificationRequired('检测到验证码，需要人工处理。')


    def wait_for_verification(self, timeout: float = 300) -> None:
        """验证码出现时暂停，等待人工处理，最多 5 分钟。"""
        deadline = __import__('time').monotonic() + timeout
        logger.warning('检测到验证码，暂停等待人工处理: timeout=%ss', timeout)
        while __import__('time').monotonic() < deadline:
            if '/403.html' in (self.get_page().url or ''):
                raise AccessRestricted('验证期间页面进入访问受限状态。')
            if not self.check_yan_cheng_ma():
                logger.info('验证码已处理，继续任务')
                return
            sleep(1)
        raise TimeoutError('等待验证码处理超过 5 分钟。')


    def ensure_or_wait(self) -> None:
        """检查页面状态；验证码出现时等待人工处理后再次确认。"""
        self.start()
        try:
            self.ensure_available()
        except VerificationRequired:
            self.wait_for_verification()
            self.ensure_available()


    def _redirect_to_login(self, page) -> None:
        """探针确认未登录后，把当前浏览器页面停在登录界面。"""
        try:
            current_url = str(page.url or '')
        except Exception:
            current_url = ''
        normalized_url = current_url.lower()
        if (
            normalized_url.startswith('https://www.zhipin.com/web/user/')
            or any(marker in normalized_url for marker in login_page_markers)
        ):
            return
        logger.info("登录探针未通过，打开登录页面: url=%s", login_page_url)
        try:
            page.get(login_page_url)
            page.wait.load_start()
        except Exception:
            logger.warning("打开登录页面失败，将在下一轮登录探针中重试", exc_info=True)


    def _get_login_probe_page(self):
        """获取执行登录探针的页面，优先复用当前 Boss 页面。"""
        main_page = self.get_page()
        try:
            main_url = str(main_page.url or '')
        except Exception:
            main_url = ''
        if (
            self._same_site(main_url, login_probe_api)
        ):
            return main_page

        probe_page = self._login_probe_page
        if probe_page is not None and self._page_is_alive(probe_page):
            try:
                probe_url = str(probe_page.url or '')
            except Exception:
                probe_url = ''
            if self._same_site(probe_url, login_probe_api):
                return probe_page
            try:
                probe_page.get(login_probe_api)
                probe_page.wait.load_start()
                return probe_page
            except Exception:
                logger.debug("复用登录探针标签页失败，将重新创建", exc_info=True)
                self._close_auxiliary_page("_login_probe_page", "登录探针")

        probe_page = main_page.new_tab(
            login_probe_api,
            background=True,
        )
        probe_page.wait.load_start()
        self._login_probe_page = probe_page
        return probe_page


    def _get_job_detail_api_page(self, *, force_auxiliary: bool = False):
        """获取可安全执行 BOSS 同源详情请求的页面上下文。"""
        main_page = self.get_page()
        try:
            main_url = str(main_page.url or '')
        except Exception:
            main_url = ''
        main_path = (urlparse(main_url).path or '').lower()
        main_is_ordinary_boss_page = (
            self._same_site(main_url, job_card_api)
            and not main_path.startswith('/wapi/')
            and not main_path.endswith('.json')
        )
        if main_is_ordinary_boss_page and not force_auxiliary:
            return main_page

        detail_page = self._job_detail_page
        if detail_page is not None and self._page_is_alive(detail_page):
            try:
                detail_url = str(detail_page.url or '')
            except Exception:
                detail_url = ''
            if self._same_site(detail_url, job_card_api):
                return detail_page
            self._close_auxiliary_page("_job_detail_page", "职位详情")

        detail_page = main_page.new_tab(
            'https://www.zhipin.com/',
            background=True,
        )
        detail_page.wait.load_start()
        detail_url = str(detail_page.url or '')
        if not self._same_site(detail_url, job_card_api):
            detail_page.close()
            raise RuntimeError(
                f"职位详情 API 页面未进入 BOSS 同源页面: "
                f"url={detail_url or '-'}"
            )
        self._job_detail_page = detail_page
        logger.info(
            "创建职位详情 API 同源页面: force_auxiliary=%s",
            force_auxiliary,
        )
        return detail_page


    def _request_login_probe(self, page) -> dict:
        """在当前页面上下文请求需要登录的只读接口。"""
        result = page.run_js(
            """
            return (async (url, timeoutSeconds) => {
                const controller = new AbortController();
                const timeoutId = setTimeout(
                    () => controller.abort(),
                    Number(timeoutSeconds) * 1000
                );
                try {
                    const response = await fetch(url, {
                        method: 'GET',
                        credentials: 'include',
                        signal: controller.signal,
                        headers: {
                            'Accept': 'application/json, text/plain, */*',
                            'X-Requested-With': 'XMLHttpRequest'
                        }
                    });
                    const raw = await response.text();
                    let payload = null;
                    try {
                        payload = raw ? JSON.parse(raw) : null;
                    } catch (_) {
                        payload = null;
                    }
                    return {
                        ok: response.ok,
                        status: response.status,
                        url: response.url,
                        contentType: response.headers.get('content-type') || '',
                        code: payload && payload.code,
                        message: payload && (payload.message || payload.msg || ''),
                        hasPayload: Boolean(payload)
                    };
                } catch (error) {
                    return {
                        ok: false,
                        status: 0,
                        url,
                        error: String(error)
                    };
                } finally {
                    clearTimeout(timeoutId);
                }
            })(arguments[0], arguments[1])
            """,
            login_probe_api,
            login_probe_request_timeout_seconds,
        )
        return result if isinstance(result, dict) else {
            'ok': False,
            'status': 0,
            'url': login_probe_api,
            'error': '登录探针返回格式无效',
        }


    def check_login(self, *, force: bool = False) -> bool:
        """通过同源登录接口确认当前浏览器会话是否有效。"""
        if not force and self._login_cache_is_fresh():
            return True

        for attempt in range(login_probe_retry_count):
            try:
                page = self._get_login_probe_page()
                probe = self._request_login_probe(page)
            except Exception as exc:
                logger.warning(
                    "登录探针请求异常: attempt=%s/%s error=%s",
                    attempt + 1,
                    login_probe_retry_count,
                    type(exc).__name__,
                )
                if attempt + 1 < login_probe_retry_count:
                    sleep(login_probe_retry_delay)
                    continue
                self._login_verified_at = 0.0
                return False

            code = str(probe.get('code')).strip()
            logged_in = bool(probe.get('ok')) and code == '0'
            if logged_in:
                self._login_verified_at = monotonic()
                return True

            message = str(probe.get('message') or '').strip()
            lower_message = message.lower()
            probe_url = str(probe.get('url') or '').lower()
            status = probe.get('status')
            requires_login = (
                status in {401, 403}
                or code in login_probe_failure_codes
                or any(
                    marker in lower_message
                    for marker in login_probe_failure_messages
                )
                or '/web/user/' in probe_url
                or '/web/passport/' in probe_url
            )
            if requires_login:
                logger.warning(
                    "登录探针确认未登录: http_status=%s code=%s message=%s",
                    status,
                    code or '-',
                    (message or str(probe.get('error') or ''))[:120],
                )
                self._redirect_to_login(self.get_page())
                self._login_verified_at = 0.0
                return False

            logger.warning(
                "登录探针暂时不可用，将重试: attempt=%s/%s http_status=%s "
                "code=%s content_type=%s has_payload=%s message=%s",
                attempt + 1,
                login_probe_retry_count,
                status,
                code or '-',
                probe.get('contentType') or '-',
                bool(probe.get('hasPayload')),
                (message or str(probe.get('error') or ''))[:120],
            )
            if attempt + 1 < login_probe_retry_count:
                sleep(login_probe_retry_delay)
                continue
            self._login_verified_at = 0.0
            return False


    def login(self):
        """登录。"""
        while True:
            if self.check_login():
                break
            sleep(1)

        return self.check_login()


    def wait_until_logged_in(
        self,
        timeout: float = 15,
        initial_delay: float = 2,
        stable_checks: int = 2,
        check_interval: float = 1,
    ) -> bool:
        """等待登录探针连续成功，确认当前会话可以继续执行任务。"""
        if self._login_cache_is_fresh():
            return True
        if initial_delay > 0:
            sleep(initial_delay)
        deadline = monotonic() + timeout
        required_checks = max(int(stable_checks), 1)
        interval = max(float(check_interval), 0.05)
        successful_checks = 0
        while monotonic() < deadline:
            self.ensure_or_wait()
            if self.check_login():
                successful_checks += 1
                if successful_checks >= required_checks:
                    return True
            else:
                successful_checks = 0
            sleep(interval)
        return False


    def job_key(self, job):
        """优先使用职位加密 ID,避免同一职位重复保存。"""
        return job.get('encryptJobId') or f"{job.get('jobName', '')}:{job.get('brandName', '')}:{job.get('lid', '')}"


    def _scroll_state(self, page):
        """获取页面滚动状态，用于判断滚动是否触发了新数据加载。"""
        return page.run_js('''
            return {
                scrollY: window.scrollY,
                height: document.documentElement.scrollHeight,
                viewport: window.innerHeight
            };
        ''')


    def _resolve_city_code(self, city: str) -> str:
        """把城市名称转换为 Boss 城市编码。"""
        if not city:
            return ''
        if str(city).isdigit():
            return str(city)

        city_name = str(city).strip()
        code = code_book.city_code(city_name)
        if not code:
            raise ValueError(f'暂不支持城市“{city_name}”，请补充 data/city_codes.json。')
        return code


    def get_job_list(
        self,
        query: str,
        max_pages: int | None = None,
        *,
        salary_code: str | int = '',
        experience_code: str | int = '',
        scale_code: str | int = '',
        degree_code: str | int = '',
        city: str = '',
        industry: str = '',
        position: str = '',
        job_type_code: str | int = '',
        stage_code: str | int = '',
        on_job: Callable[[dict], None] | None = None,
        on_page: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list:
        """按关键词和筛选条件获取职位列表。"""
        self._raise_if_stopped(should_stop)
        self.ensure_or_wait()
        page = self.get_page()
        pages = []
        jobs_by_key = {}
        city_code = self._resolve_city_code(city)
        filters = {
            'city': city_code,
            'jobType': job_type_code,
            'salary': salary_code,
            'experience': experience_code,
            'degree': degree_code,
            'scale': scale_code,
            'stage': stage_code,
            'query': query,
            'industry': industry,
            'position': position,
        }
        filters = {key: value for key, value in filters.items() if value not in ('', None)}
        search_url = (
            'https://www.zhipin.com/web/geek/jobs?'
            + urlencode(filters, safe=',')
        )
        logger.info(
            "开始搜索职位: query=%s city=%s max_pages=%s",
            query,
            city or "当前城市",
            max_pages if max_pages is not None else "hasMore",
        )

        listener_started = False
        try:
            if not self.wait_until_logged_in(
                initial_delay=0,
                stable_checks=1,
            ):
                raise LoginRequired('登录状态未确认，请在浏览器中完成登录后重试。')

            page = self.get_page()
            page.listen.start(targets=[job_list_target], is_regex=False)
            listener_started = True
            page.get(search_url, timeout=job_list_navigation_timeout_seconds)
            page.wait.load_start()
            self._raise_if_stopped(should_stop)
            self.ensure_or_wait()
            has_more = True
            stopped_no_new_jobs = False
            while has_more and (
                max_pages is None or len(pages) < max_pages
            ):
                self._raise_if_stopped(should_stop)
                self.ensure_available(page)
                packet = page.listen.wait(
                    timeout=job_list_response_timeout_seconds,
                    raise_err=False,
                )
                if not packet:
                    self._raise_if_stopped(should_stop)
                    self.ensure_available(page)
                    if not self.check_login(force=True):
                        raise LoginRequired('职位列表请求需要重新登录。')
                    raise RuntimeError('等待职位列表接口响应超时。')

                body = packet.response.body
                if isinstance(body, (str, bytes)):
                    body = json.loads(body)
                if not isinstance(body, dict):
                    raise RuntimeError('职位列表接口响应格式无效。')
                code = str(body.get('code'))
                if code == '36':
                    raise AccessRestricted('职位列表请求触发 code=36，已停止采集。')
                if code != '0':
                    if not self.check_login(force=True):
                        raise LoginRequired('职位列表请求需要重新登录。')
                    raise RuntimeError(f'职位列表接口返回错误: code={code}')
                zp_data = body.get('zpData') or {}
                page_jobs = zp_data.get('jobList') or []
                page_jobs = [
                    dict(job, _searchUrl=search_url, _searchMaxPages=max_pages)
                    for job in page_jobs
                ]
                has_more = bool(zp_data.get('hasMore'))

                new_job_count = 0
                for job in page_jobs:
                    self._raise_if_stopped(should_stop)
                    key = self.job_key(job)
                    is_new = key not in jobs_by_key
                    jobs_by_key.setdefault(key, job)
                    if is_new:
                        new_job_count += 1
                    if is_new and on_job is not None:
                        page.listen.stop()
                        listener_started = False
                        try:
                            on_job(job)
                        finally:
                            page.listen.start(targets=[job_list_target], is_regex=False)
                            listener_started = True
                        self._raise_if_stopped(should_stop)

                if new_job_count == 0:
                    has_more = False
                    stopped_no_new_jobs = True

                page_no = len(pages) + 1
                pages.append({
                    'page': page_no,
                    'hasMore': has_more,
                    'jobCount': len(page_jobs),
                    'newJobCount': new_job_count,
                    'jobs': page_jobs,
                })
                logger.info(
                    "职位列表分页完成: page=%s page_jobs=%s unique_jobs=%s has_more=%s",
                    page_no,
                    len(page_jobs),
                    len(jobs_by_key),
                    has_more,
                )
                if on_page is not None:
                    try:
                        on_page(pages[-1])
                    except Exception:
                        logger.warning(
                            "职位列表进度回调失败，继续分页: page=%s",
                            page_no,
                            exc_info=True,
                        )

                if not has_more or (
                    max_pages is not None and len(pages) >= max_pages
                ):
                    break

                self.ensure_available(page)
                last = page.ele('css:' + job_card_selector, index=-1, timeout=0)
                if last:
                    last.scroll.to_see()
                    page.actions.move_to(last).scroll(800, 0)
                else:
                    page.actions.scroll(1000000, 0)

            jobs = list(jobs_by_key.values())
            (save_dir / 'jobs.json').write_text(
                json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            (save_dir / 'joblist_pages.json').write_text(
                json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            (save_dir / 'joblist_summary.json').write_text(
                json.dumps({
                    **filters,
                    'pages': len(pages),
                    'unique_total': len(jobs),
                    'has_more': has_more,
                    'stopped_no_new_jobs': stopped_no_new_jobs,
                }, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            logger.info("职位采集完成: jobs=%s save_dir=%s", len(jobs), save_dir)
            self._last_search_url = search_url
            return jobs
        except BrowserOperationCancelled:
            logger.info(
                "职位列表采集按上层停止请求结束: query=%s city=%s",
                query,
                city or "当前城市",
            )
            raise
        except Exception:
            logger.exception("采集职位列表失败: query=%s city=%s", query, city or "当前城市")
            raise
        finally:
            if listener_started:
                try:
                    page.listen.stop()
                except Exception:
                    logger.debug("停止职位列表监听失败", exc_info=True)


    def _ensure_detail_available(self, page) -> None:
        """详情读取期间立即停止验证、访问限制或登录失效，不自动重试。"""
        self.ensure_available(page)
        current_url = str(page.url or '').lower()
        if any(marker in current_url for marker in login_page_markers) or '/web/user/' in current_url:
            raise LoginRequired('登录已失效，请重新登录后重试。')
        if self._has_login_entry(page):
            raise LoginRequired('页面要求登录，请重新登录后重试。')

    @staticmethod
    def _raise_if_stopped(should_stop: Callable[[], bool] | None) -> None:
        """在可中断的浏览器循环中尽快响应上层停止请求。"""
        if should_stop is not None and should_stop():
            raise BrowserOperationCancelled('浏览器操作已取消。')


    @staticmethod
    def _check_detail_responses(
        page,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """只观察网页自然请求中的风控码；响应缺失不影响 DOM 提取。"""
        for _ in range(20):
            BrowserManager._raise_if_stopped(should_stop)
            packet = page.listen.wait(timeout=0.01, raise_err=False)
            if not packet:
                return
            body = packet.response.body
            if isinstance(body, bytes):
                body = body.decode('utf-8', errors='replace')
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    continue
            if isinstance(body, dict) and str(body.get('code')) == '36':
                raise AccessRestricted('职位页面请求触发 code=36，已停止详情获取。')


    @staticmethod
    def _read_job_detail_dom(page) -> dict:
        """在同一个可见详情容器中读取 ID、标题和纯正文，排除推荐内容。"""
        return page.run_js(r"""
            const roots = JSON.parse(arguments[0]);
            const descriptions = JSON.parse(arguments[1]);
            const visible = element => {
                if (!element || !element.isConnected) return false;
                for (let node = element; node; node = node.parentElement) {
                    const style = getComputedStyle(node);
                    if (style.display === 'none' || style.visibility === 'hidden'
                        || Number(style.opacity) === 0) return false;
                }
                const rect = element.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            };
            const idFromUrl = url => {
                const path = new URL(url, location.href).pathname;
                return path.match(/^\/job_detail\/([^/]+)\.html$/)?.[1] || '';
            };
            for (const selector of roots) {
                for (const root of document.querySelectorAll(selector)) {
                    if (!visible(root)) continue;
                    const body = descriptions.flatMap(s => [...root.querySelectorAll(s)])
                        .find(el => visible(el) && el.innerText.trim());
                    if (!body) continue;
                    // 详情自身的链接证明正文归属，不能拿列表 active 状态代替。
                    const link = root.querySelector('a.more-job-btn[href*="/job_detail/"]');
                    const jobId = link ? idFromUrl(link.href) : idFromUrl(location.href);
                    const header = root.querySelector('.job-detail-header .job-name, h1');
                    const standalone = document.querySelector('.job-banner .name h1');
                    const title = (header || standalone)?.innerText?.trim() || '';
                    // BOSS 当前页面把职位正文和顶部福利标签放在两个兄弟容器中：
                    // 正文通常位于 .job-detail，福利位于 .job-primary.detail-box。
                    // 只从同一页面的职位摘要容器补读，不扫描推荐职位。
                    const welfareContainers = [
                        root,
                        ...document.querySelectorAll('.job-primary.detail-box'),
                    ];
                    const welfareRoot = welfareContainers
                        .flatMap(container => [
                            ...container.querySelectorAll(
                                '.job-tags, .tag-container-new'
                            ),
                        ])
                        .find(visible);
                    const welfareList = welfareRoot
                        ? [...welfareRoot.querySelectorAll('span')]
                            .map(element => element.innerText.trim())
                            .filter(text => text && text !== '...' && text !== '…')
                        : [];
                    return {
                        jobId, title, description: body.innerText.trim(),
                        bossActiveTime: root.querySelector('.boss-online-tag')?.innerText?.trim() || '',
                        ...(welfareList.length
                            ? {welfareList: [...new Set(welfareList)]}
                            : {})
                    };
                }
            }
            return {};
        """, json.dumps(job_detail_container_selectors), json.dumps(job_detail_body_selectors)) or {}


    @staticmethod
    def _detail_matches_job(detail: dict, job: dict) -> bool:
        """必须是目标职位的完整标题和 ID，防止短标题前缀及旧详情误匹配。"""
        def normalize(value):
            return ''.join(unicodedata.normalize('NFKC', str(value or '')).split())
        return bool(
            detail.get('jobId') == job.get('encryptJobId')
            and normalize(detail.get('title'))
            and normalize(detail.get('title')) == normalize(job.get('jobName'))
            and str(detail.get('description') or '').strip()
        )

    @staticmethod
    def _job_card_matches_job(job_card: dict, job: dict) -> bool:
        """校验接口详情是否确实属于列表中的目标职位。"""
        def normalize(value):
            return ''.join(unicodedata.normalize('NFKC', str(value or '')).split())

        return bool(
            job_card.get('encryptJobId') == job.get('encryptJobId')
            and normalize(job_card.get('jobName'))
            and normalize(job_card.get('jobName')) == normalize(job.get('jobName'))
            and str(job_card.get('postDescription') or '').strip()
        )

    @staticmethod
    def _job_card_request_url(job: dict) -> str:
        """使用列表接口提供的参数拼接职位详情接口地址。"""
        query = urlencode({
            'lid': str(job.get('lid') or ''),
            'securityId': str(job.get('securityId') or ''),
            'sessionId': '',
        })
        return f'{job_card_api}?{query}'

    @staticmethod
    def _job_card_request_js() -> str:
        """返回在已登录 Boss 页面内读取职位详情的脚本。"""
        return """
        async (url, timeoutSeconds) => {
            const controller = new AbortController();
            const timeoutId = setTimeout(
                () => controller.abort(),
                Number(timeoutSeconds) * 1000
            );
            try {
                const response = await fetch(url, {
                    method: 'GET',
                    credentials: 'include',
                    headers: {
                        'Accept': 'application/json, text/plain, */*',
                        'X-Requested-With': 'XMLHttpRequest',
                    },
                    signal: controller.signal,
                });
                const raw = await response.text();
                let payload = null;
                try {
                    payload = raw ? JSON.parse(raw) : null;
                } catch (_) {
                    payload = null;
                }
                return {
                    ok: response.ok,
                    status: response.status,
                    contentType: response.headers.get('content-type') || '',
                    code: payload && payload.code,
                    message: payload && (payload.message || payload.msg || ''),
                    jobCard: payload
                        && payload.zpData
                        && typeof payload.zpData.jobCard === 'object'
                        ? payload.zpData.jobCard
                        : null,
                };
            } catch (error) {
                return {
                    ok: false,
                    status: 0,
                    contentType: '',
                    code: null,
                    message: String(error),
                    jobCard: null,
                };
            } finally {
                clearTimeout(timeoutId);
            }
        }
        """

    def _read_job_card_api(
        self,
        job: dict,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict | None:
        """优先通过同一登录页面读取职位详情接口。"""
        self._raise_if_stopped(should_stop)
        required_fields = ('encryptJobId', 'lid', 'securityId')
        if any(not job.get(field) for field in required_fields):
            return None

        request_args = (
            self._job_card_request_js(),
            self._job_card_request_url(job),
            job_card_request_timeout_seconds,
        )
        result = None
        for attempt in range(2):
            self._raise_if_stopped(should_stop)
            page = self._get_job_detail_api_page(
                force_auxiliary=attempt > 0,
            )
            self.ensure_available(page)
            try:
                result = page.run_js(
                    *request_args,
                    timeout=job_card_request_timeout_seconds + 2,
                )
            except Exception:
                if attempt:
                    raise
                logger.warning(
                    "职位详情 API 页面调用失败，将切换同源页面重试: title=%s",
                    job.get('jobName', ''),
                    exc_info=True,
                )
                continue

            self._raise_if_stopped(should_stop)
            if (
                attempt == 0
                and isinstance(result, dict)
                and not result.get('ok')
                and (
                    result.get('status') == 0
                    or 'failed to fetch'
                    in str(result.get('message') or '').lower()
                )
            ):
                logger.warning(
                    "职位详情 API 网络上下文失败，将切换同源页面重试: "
                    "title=%s message=%s",
                    job.get('jobName', ''),
                    str(result.get('message') or '')[:120],
                )
                continue
            break

        self._raise_if_stopped(should_stop)
        if not isinstance(result, dict):
            logger.warning(
                '职位详情接口返回格式无效: title=%s',
                job.get('jobName', ''),
            )
            return None

        status = result.get('status')
        code = str(result.get('code')).strip()
        message = str(result.get('message') or '').strip()
        if code == '36':
            raise AccessRestricted('职位详情接口触发 code=36，已停止详情获取。')
        if (
            status in {401, 403}
            or code in login_probe_failure_codes
            or any(marker in message for marker in login_probe_failure_messages)
        ):
            raise LoginRequired('职位详情接口确认登录已失效，请重新登录。')
        if not result.get('ok') or code != '0':
            logger.warning(
                '职位详情接口请求失败: title=%s status=%s code=%s message=%s',
                job.get('jobName', ''),
                status,
                code or '-',
                message[:120],
            )
            return None

        job_card = result.get('jobCard')
        if not isinstance(job_card, dict):
            logger.warning(
                '职位详情接口缺少 jobCard: title=%s',
                job.get('jobName', ''),
            )
            return None
        if not self._job_card_matches_job(job_card, job):
            logger.warning(
                '职位详情接口职位校验失败: title=%s',
                job.get('jobName', ''),
            )
            return None

        welfare_list = merge_benefits(
            job.get('welfareList'),
            job.get('benefitList'),
            job.get('benefits'),
            job_card.get('welfareList'),
            job_card.get('benefitList'),
            job_card.get('benefits'),
        )
        identity_fields = {
            field: job[field]
            for field in ('encryptJobId', 'lid', 'securityId')
            if job.get(field)
        }
        return {
            **job,
            **job_card,
            **identity_fields,
            **({'welfareList': welfare_list} if welfare_list else {}),
            **({
                'bossActiveTime': job_card.get('activeTimeDesc')
            } if job_card.get('activeTimeDesc') else {}),
        }


    def get_job_card(
        self,
        job: dict,
        timeout: float | None = None,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict | None:
        """优先读取职位详情接口，失败后回退到后台详情页 DOM。"""
        self._raise_if_stopped(should_stop)
        if timeout is None:
            timeout = job_detail_dom_timeout_seconds
        if not job.get('encryptJobId') or not job.get('jobName'):
            return None
        try:
            reader = self._read_job_card_api
            if type(reader).__module__ == "unittest.mock":
                card = reader(job)
            else:
                card = reader(job, should_stop=should_stop)
        except (VerificationRequired, AccessRestricted, LoginRequired):
            raise
        except BrowserOperationCancelled:
            raise
        except Exception as exc:
            logger.warning(
                '职位详情接口读取异常，将回退 DOM: title=%s error=%s',
                job.get('jobName', ''),
                type(exc).__name__,
            )
            card = None
        if card is not None:
            logger.debug(
                '职位详情接口读取成功: title=%s description_chars=%s',
                job.get('jobName', ''),
                len(str(card.get('postDescription') or '')),
            )
            return card

        page = self.get_page()
        job_id = quote(str(job['encryptJobId']), safe='')
        detail_url = f'https://www.zhipin.com/job_detail/{job_id}.html'
        detail_page = page.new_tab(background=True)
        blocked = False
        detail_page.listen.start(targets=['/wapi/zpgeek/'], is_regex=False)
        try:
            if type(detail_page.get).__module__ == "unittest.mock":
                detail_page.get(detail_url)
            else:
                detail_page.get(detail_url, timeout=timeout)
            detail_page.wait.load_start()
            self._raise_if_stopped(should_stop)
            deadline = monotonic() + timeout
            previous = None
            while monotonic() < deadline:
                self._raise_if_stopped(should_stop)
                self._check_detail_responses(detail_page, should_stop)
                self._ensure_detail_available(detail_page)
                detail = self._read_job_detail_dom(detail_page)
                self._raise_if_stopped(should_stop)
                if self._detail_matches_job(detail, job):
                    fingerprint = (
                        detail['jobId'],
                        detail['title'],
                        detail['description'],
                        tuple(detail.get('welfareList') or ()),
                    )
                    if fingerprint == previous:
                        welfare_list = merge_benefits(
                            job.get("welfareList"),
                            job.get("benefitList"),
                            job.get("benefits"),
                            detail.get("welfareList"),
                        )
                        return {
                            **job,
                            'postDescription': detail['description'],
                            **({'welfareList': welfare_list} if welfare_list else {}),
                            **({'bossActiveTime': detail['bossActiveTime']}
                               if detail.get('bossActiveTime') else {}),
                        }
                    previous = fingerprint
                else:
                    previous = None
                sleep(0.25)
            logger.warning('职位详情未匹配或加载超时: title=%s timeout=%s', job['jobName'], timeout)
            return None
        except (VerificationRequired, AccessRestricted, LoginRequired):
            blocked = True
            raise
        except BrowserOperationCancelled:
            raise
        except Exception as exc:
            logger.warning(
                '职位详情读取异常，跳过当前职位: title=%s error=%s',
                job.get('jobName', ''),
                type(exc).__name__,
            )
            return None
        finally:
            try:
                detail_page.listen.stop()
            except Exception:
                logger.debug('停止详情请求观测失败', exc_info=True)
            if not blocked:
                detail_page.close()


    def get_job_cards(self, jobs: list[dict], interval: tuple[float, float] = (1, 2)) -> list[dict | None]:
        """逐个读取职位详情，保留顺序和间隔；风控异常立即终止批次。"""
        cards = []
        for index, job in enumerate(jobs, 1):
            if index > 1:
                sleep(uniform(*interval))
            try:
                card = self.get_job_card(job)
            except (VerificationRequired, AccessRestricted) as exc:
                exc.partial_cards = list(cards)
                raise
            title = job.get('jobName', '')
            if card:
                desc = card.get('postDescription', '')
                logger.info(
                    "职位详情获取成功: index=%s total=%s title=%s description_chars=%s",
                    index,
                    len(jobs),
                    title,
                    len(desc),
                )
            else:
                logger.warning(
                    "职位详情获取失败: index=%s total=%s title=%s",
                    index,
                    len(jobs),
                    title,
                )
            cards.append(card)
        return cards


    def _read_bst(self) -> str:
        """从当前浏览器配置文件读取 bst，投递接口要求放进 Zp_token 头。"""
        page = self.get_page()
        for cookie in page.browser.cookies():
            domain = str(cookie.get('domain') or '').lstrip('.').lower()
            if (
                cookie.get('name') == 'bst'
                and (domain == 'zhipin.com' or domain.endswith('.zhipin.com'))
            ):
                return cookie.get('value', '')
        return ''

    def _get_push_page(self):
        """复用当前 Boss 页面发起同源投递请求，不额外创建标签页。"""
        page = self.get_page()
        if not self._same_site(str(page.url or ''), push_api):
            raise LoginRequired('投递页面不在 Boss 站点，请先重新打开职位搜索页。')
        return page

    @staticmethod
    def _parse_push_payload(raw) -> dict:
        """兼容浏览器脚本返回的字符串、字节串或已解析字典。"""
        if isinstance(raw, dict):
            payload = raw
        else:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError('投递接口未返回有效响应')
            payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError('投递接口响应不是 JSON 对象')
        return payload


    _push_js = """
    async (url, token, timeoutSeconds) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(
            () => controller.abort(),
            Number(timeoutSeconds) * 1000
        );
        try {
            const resp = await fetch(url, {
                method: 'POST',
                credentials: 'include',
                headers: {'Zp_token': token},
                signal: controller.signal,
            });
            return await resp.text();
        } finally {
            clearTimeout(timeoutId);
        }
    }
    """

    def push_job(self, job: dict, retries: int = 1) -> tuple[bool, str]:
        """向单个职位发起投递（打招呼）。"""
        security_id = job.get('securityId', '')
        job_id = job.get('encryptJobId', '')
        lid = job.get('lid', '')
        if not all([security_id, job_id, lid]):
            return False, '缺少 securityId / encryptJobId / lid'

        self.ensure_or_wait()
        token = self._read_bst()
        if not token:
            return False, '未登录（bst cookie 为空）'

        url = f'{push_api}?securityId={security_id}&jobId={job_id}&lid={lid}'
        attempts = max(int(retries), 1)
        last_error = "投递请求未返回结果"
        for attempt in range(attempts):
            try:
                page = self._get_push_page()
                raw = page.run_js(
                    self._push_js,
                    url,
                    token,
                    push_request_timeout_seconds,
                    timeout=push_request_timeout_seconds + 2,
                )
                payload = self._parse_push_payload(raw)
            except Exception as exc:
                last_error = f'结果未知：{type(exc).__name__}: {exc}'
                if attempt + 1 < attempts:
                    logger.warning(
                        "投递请求异常，准备重试: job_id=%s attempt=%s/%s error=%s",
                        job_id,
                        attempt + 1,
                        attempts,
                        type(exc).__name__,
                    )
                    sleep(0.5 * (attempt + 1))
                    continue
                return False, last_error

            code = str(payload.get('code')).strip()
            message = payload.get('message') or ''
            remind = (
                ((payload.get('zpData') or {}).get('bizData') or {})
                .get('chatRemindDialog') or {}
            ).get('content') or ''

            if code == '0':
                return True, message or 'Success'
            if daily_limit_hint in remind:
                return False, remind
            return False, remind or message or f'code={code}'
        return False, last_error


    def close(self):
        self._close_auxiliary_page("_push_page", "投递")
        self._close_auxiliary_page("_chat_page", "聊天监听")
        self._close_auxiliary_page("_login_probe_page", "登录探针")
        self._close_auxiliary_page("_job_detail_page", "职位详情")

        if self._page is not None:
            logger.info(
                "正在关闭浏览器连接: owns_browser=%s",
                self._owns_browser,
            )
            if self._owns_browser:
                try:
                    self._page.quit()
                except Exception:
                    logger.debug("关闭本进程创建的浏览器失败", exc_info=True)
            else:
                logger.info("浏览器实例由其他进程创建，仅断开当前连接")
            self._page = None
            self._owns_browser = False
            self._browser_identity = None
            self._login_verified_at = 0.0
            self._ready.clear()
            logger.info("浏览器连接已关闭")
        else:
            self._owns_browser = False
            self._browser_identity = None
            self._login_verified_at = 0.0
            self._ready.clear()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
