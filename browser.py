import json
import pathlib
import unicodedata
from dataclasses import dataclass
from threading import Event, RLock
from random import uniform
from typing import Callable
from time import monotonic, sleep
from urllib.request import urlopen
from urllib.parse import parse_qs, quote, urlencode, urlparse

from DrissionPage import Chromium, ChromiumPage, ChromiumOptions

from matcher import code_book
from logging_config import get_logger

logger = get_logger(__name__)

# 所有任务共享这把锁，避免并发任务同时操作同一个浏览器页面。
BROWSER_IO_LOCK = RLock()


class BrowserBlocked(Exception):
    """浏览器操作被阻止；批量详情读取时携带已完成的结果。"""

    def __init__(self, message: str):
        super().__init__(message)
        # 保留失败位置的 None，调用方仍能按原职位顺序对应已处理结果。
        self.partial_cards: list[dict | None] = []


class VerificationRequired(BrowserBlocked):
    """需要人工处理验证码。"""


class AccessRestricted(BrowserBlocked):
    """页面访问受限。"""


class LoginRequired(BrowserBlocked):
    """登录态缺失或已失效，需要用户重新登录。"""


base_dir = pathlib.Path(__file__).resolve().parent
user_data_dir = base_dir.parent / 'user_data' / 'user_data'
save_dir = base_dir / 'jobs_data'
user_data_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)

job_list_target = '/wapi/zpgeek/search/joblist.json'
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
push_api = 'https://www.zhipin.com/wapi/zpgeek/friend/add.json'
daily_limit_hint = '您今天已与120位BOSS沟通'
max_joblist_pages = 8
chat_url = 'https://www.zhipin.com/web/geek/chat'

# 职位详情在后台标签页打开网页后读取 DOM，不触碰搜索列表页；
# 这里保留多个选择器以兼容 BOSS 页面不同版本的详情容器。
job_card_selector = '.job-card-wrapper, .job-card-wrap'
job_detail_body_selectors = (
    '.job-detail-body .desc',
    '.job-sec-text',
    '.job-description',
)
job_detail_container_selectors = (
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

# 浏览器调试端口。DrissionPage 4.x 默认使用 9222，该端口常被其他程序
# （如残留的旧 Chrome 实例、其他自动化脚本）占用，导致连接失败。
# 这里改用 49152-65535 私有/动态端口段中偏后的端口，避免冲突。
debug_port = 49613


def _debug_pages(port: int) -> list[dict]:
    """读取 DevTools 页面列表；浏览器尚未就绪或连接失败时返回空列表。"""
    try:
        with urlopen(f'http://127.0.0.1:{port}/json/list', timeout=1) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


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


@dataclass
class JobConfig:
    job_list_target = '/wapi/zpgeek/search/joblist.json'
    max_joblist_pages = 8




class BrowserManager:
    def __init__(self):
        self.user_data_dir = user_data_dir
        self._page:ChromiumPage|None = None
        self._owns_browser = False
        self._browser_identity: str | None = None
        # 投递请求使用独立的后台标签页，避免影响用户正在查看的搜索页面。
        self._push_page = None
        self._last_search_url = ''
        # 浏览器就绪信号：_page 首次可用时置位，供聊天监听等后台线程等待。
        self._ready = Event()
        self._start_lock = RLock()


    def get_page(self) -> ChromiumPage:
        if self._page is None:
            raise ValueError("Browser not started")
        return self._page


    def start(self):
        """优先复用或重连现有实例；仅在 DevTools 不存在时创建浏览器。"""
        with self._start_lock:
            if self._page is not None and self._page_is_alive(self._page):
                return self

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
                    self._push_page = None
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
                    # 端点可能在探测后关闭；只有确认实例仍存在时才报告重连错误。
                    if _debug_browser(debug_port) is not None:
                        logger.exception("重连现有浏览器失败: port=%s", debug_port)
                        raise
                    logger.info("浏览器实例已退出，将启动新实例")

            logger.info("正在创建浏览器实例: user_data_dir=%s port=%s", self.user_data_dir, debug_port)
            _stop_stale_browser(debug_port, self.user_data_dir)
            self._page = ChromiumPage(addr_or_opts=options)
            self._push_page = None
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
        """连接现有 DevTools 实例，重连原标签或在没有标签时创建一个。"""
        browser = Chromium(addr_or_opts=options)
        try:
            tab_ids = browser.tab_ids
        except Exception:
            try:
                browser.reconnect()
            except Exception:
                # 进程级缓存可能还留着已经断开的 Chromium 对象。
                Chromium._BROWSERS.pop(browser.id, None)
                browser = Chromium(addr_or_opts=options)
            tab_ids = browser.tab_ids

        if self._page is not None:
            try:
                page_id = self._page.tab_id
                if page_id in tab_ids:
                    self._page.reconnect()
                    if self._page_is_alive(self._page):
                        return self._page
            except Exception:
                logger.info("原浏览器标签无法重连，将选择其他标签")

        if tab_ids:
            page_id = tab_ids[0]
        else:
            page_id = browser.new_tab(background=False).tab_id

        cached_page = ChromiumPage._PAGES.get(browser.id)
        if cached_page is not None and cached_page.tab_id == page_id:
            if self._page_is_alive(cached_page):
                return cached_page
            ChromiumPage._PAGES.pop(browser.id, None)
            cached_page = None
        if cached_page is None:
            # DrissionPage 4.1.1.4 按 browser id 缓存 ChromiumPage；
            # 目标标签变化时清掉旧页面缓存，避免继续返回断开的 Page 对象。
            ChromiumPage._PAGES.pop(browser.id, None)
            page = ChromiumPage(addr_or_opts=options, tab_id=page_id)
        else:
            page = cached_page
        if not self._page_is_alive(page):
            try:
                page.reconnect()
            except Exception:
                ChromiumPage._PAGES.pop(browser.id, None)
                page = ChromiumPage(addr_or_opts=options, tab_id=page_id)
        if not self._page_is_alive(page):
            raise RuntimeError("现有浏览器的页面连接仍不可用。")
        return page


    def is_ready(self) -> bool:
        """浏览器是否已经启动并持有可用页面。"""
        return self._page is not None and self._page_is_alive(self._page)


    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """等待浏览器就绪。

        返回 True 表示已就绪；超时返回 False。传入 timeout=None 表示一直等待，
        适用于后台监听线程在浏览器尚未启动时保持待命。
        """
        if self.is_ready():
            return True
        return self._ready.wait(timeout) and self.is_ready()


    def goto(self, url: str):
        logger.info("打开页面: url=%s", url)
        result = self.start().get_page().get(url)
        self.ensure_or_wait()
        return result

    def open_chat(self) -> ChromiumPage:
        """打开聊天页，复用当前浏览器登录态。"""
        page = self.goto(chat_url)
        page.wait.load_start()
        return page

    def start_chat_listener(self) -> None:
        """通过 BOSS 聊天 SDK 监听实时消息。

        BOSS 聊天页使用 geek-chat SDK，消息由 WebSocket/MQTT 通道推送。
        这里在页面上下文中注册 SDK 事件监听器，避免自行复制登录凭证或伪造协议。
        """
        page = self.open_chat()
        page.run_js(
            """
            (() => {
                if (window.__bossAgentChatBridge) return 'already_started';
                const queue = [];
                const append = (kind, payload) => {
                    try {
                        queue.push({kind, payload: JSON.parse(JSON.stringify(payload))});
                    } catch (_) {
                        queue.push({kind, payload: String(payload)});
                    }
                    if (queue.length > 200) queue.splice(0, queue.length - 200);
                };
                const sdk = window.ChatWebsocket;
                if (!sdk || typeof sdk.on !== 'function') {
                    throw new Error('聊天 SDK 尚未初始化，请确认聊天页已加载完成');
                }
                sdk.on('message', payload => append('message', payload));
                sdk.on('messageArrived', payload => append('messageArrived', payload));
                sdk.on('messageSync', payload => append('messageSync', payload));
                window.__bossAgentChatBridge = {
                    queue,
                    sdk,
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
            page = self.get_page()
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
                    message = self._normalize_chat_message(item, event.get('kind', ''))
                    if message:
                        messages.append(message)
            if messages or __import__('time').monotonic() >= deadline:
                return messages
            sleep(0.1)

    @staticmethod
    def _normalize_chat_message(payload: object, event_kind: str = '') -> dict | None:
        """兼容 SDK 的 message 和 messageArrived 两种事件结构。"""
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

        message_id = message.get('mid') or message.get('messageId') or message.get('id')
        from_user = message.get('from') if isinstance(message.get('from'), dict) else {}
        to_user = message.get('to') if isinstance(message.get('to'), dict) else {}
        sender_id = message.get('fromId') or from_user.get('uid') or from_user.get('userId')
        target_id = message.get('toId') or to_user.get('uid') or to_user.get('userId')
        friend_source = (
            message.get('friendSource')
            or message.get('fromSource')
            or from_user.get('source')
            or 0
        )
        conversation = {
            'uid': sender_id if sender_id else target_id,
            'friendSource': friend_source,
            'encryptUid': message.get('encryptBossId') or from_user.get('encryptUid', ''),
            'groupId': message.get('groupId') or '',
            'encryptGid': message.get('encryptGid') or '',
        }
        conversation_id = (
            message.get('uniqueId')
            or message.get('conversationId')
            or conversation.get('groupId')
            or f"{conversation['uid']}-{friend_source}"
        )
        if not message_id or not conversation_id:
            return None

        current_user_id = None
        try:
            current_user_id = message.get('currentUserId')
        except AttributeError:
            pass
        is_self = bool(message.get('isSelf')) or (
            current_user_id is not None and str(sender_id) == str(current_user_id)
        )
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
        """调用当前页面 SDK 发送文本消息。"""
        if not content.strip():
            raise ValueError('消息内容不能为空')
        page = self.get_page()
        result = page.run_js(
            """
            (args => {
                const sdk = window.ChatWebsocket;
                if (!sdk || typeof sdk.sendTextMessage !== 'function') {
                    throw new Error('聊天 SDK 未准备好');
                }
                const user = {
                    uid: args.conversation.uid,
                    friendSource: args.conversation.friendSource || 0,
                    encryptUid: args.conversation.encryptUid || '',
                    groupId: args.conversation.groupId || '',
                    encryptGid: args.conversation.encryptGid || '',
                };
                return sdk.sendTextMessage(user, args.content);
            })(arguments[0]);
            """,
            {'conversation': conversation, 'content': content},
        )
        return result if isinstance(result, dict) else {'result': result}

    def stop_chat_listener(self) -> None:
        """移除页面侧聊天事件桥接；不影响 BOSS 页面自身连接。"""
        if self._page is None:
            return
        try:
            self._page.run_js('window.__bossAgentChatBridge = null; return true;')
        except Exception:
            logger.debug('清理聊天监听桥接失败', exc_info=True)


    def __call__(self, url: str) ->ChromiumPage:
        return self.goto(url)


    def check_yan_cheng_ma(self, page=None) -> bool:
        """只检查当前页面是否出现可见的验证码组件。"""
        page = page if page is not None else self.get_page()
        if bool(page.run_js("""
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
        """)):
            return True

        current_url = page.url or ''
        if any(path in current_url for path in ('/403.html', '/web/passport/zp/verify.html')):
            logger.warning("检测到验证或访问限制页面: path=%s", urlparse(current_url).path)
            return True

        if bool(page.run_js("""
            const text = document.body ? document.body.innerText : '';
            const restrictedTexts = [
                '访问受限',
                '您的IP存在异常行为',
                '您的 IP 存在异常行为',
                '请勿频繁提交刷新请求',
                '您的账户存在异常行为'
            ];
            return restrictedTexts.some(item => text.includes(item));
        """)):
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


    def _request_login_probe(self, page) -> dict:
        """在当前页面上下文请求需要登录的只读接口。

        不返回用户信息或 Token，只保留登录判断所需的响应元数据，避免把
        个人信息写入日志或跨出浏览器上下文。
        """
        result = page.run_js(
            """
            (async (url) => {
                try {
                    const response = await fetch(url, {
                        method: 'GET',
                        credentials: 'include',
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
                        code: payload && payload.code,
                        message: payload && (payload.message || payload.msg || '')
                    };
                } catch (error) {
                    return {
                        ok: false,
                        status: 0,
                        url,
                        error: String(error)
                    };
                }
            })(arguments[0])
            """,
            login_probe_api,
        )
        return result if isinstance(result, dict) else {
            'ok': False,
            'status': 0,
            'url': login_probe_api,
            'error': '登录探针返回格式无效',
        }


    def check_login(self) -> bool:
        """通过需要登录的只读接口确认当前浏览器会话是否有效。"""
        page = self.get_page()
        try:
            probe = self._request_login_probe(page)
        except Exception as exc:
            logger.warning(
                "登录探针请求异常: error=%s",
                type(exc).__name__,
            )
            return False

        code = str(probe.get('code')).strip()
        logged_in = bool(probe.get('ok')) and code == '0'
        if logged_in:
            return True

        message = str(probe.get('message') or '').strip()
        lower_message = message.lower()
        probe_url = str(probe.get('url') or '').lower()
        requires_login = (
            probe.get('status') in {401, 403}
            or code in login_probe_failure_codes
            or any(
                marker in lower_message
                for marker in login_probe_failure_messages
            )
            or '/web/user/' in probe_url
            or '/web/passport/' in probe_url
        )
        logger.warning(
            "登录探针未通过: http_status=%s code=%s message=%s requires_login=%s",
            probe.get('status'),
            code or '-',
            (message or str(probe.get('error') or ''))[:120],
            requires_login,
        )
        if requires_login:
            self._redirect_to_login(page)
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
        max_pages: int = max_joblist_pages,
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
    ) -> list:
        """按关键词和筛选条件获取职位列表。"""
        self.ensure_or_wait()
        page = self.get_page()
        pages = []
        jobs_by_key = {}
        city_code = self._resolve_city_code(city)
        # URL 参数只保留有值的筛选项；筛选名称到编码的转换由调用方
        # 依据 data/*_codes.json 完成，这里只负责拼接 URL。
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
            max_pages,
        )

        listener_started = False
        try:
            if not self.wait_until_logged_in():
                raise LoginRequired('登录状态未确认，请在浏览器中完成登录后重试。')

            # 通过 URL 参数重新加载筛选条件，避免登录页完成后继续使用旧页面状态。
            page = self.get_page()
            page.listen.start(targets=[job_list_target], is_regex=False)
            listener_started = True
            page.get(search_url)
            page.wait.load_start()
            sleep(2)
            self.ensure_or_wait()
            if not self.check_login():
                raise LoginRequired('打开职位搜索页后登录状态已失效，请重新登录。')
            has_more = True
            while has_more and len(pages) < max_pages:
                self.ensure_available(page)
                if not self.check_login():
                    raise LoginRequired('职位搜索过程中登录状态已失效，请重新登录。')
                packet = page.listen.wait(timeout=10, raise_err=False)
                if not packet:
                    self.ensure_available(page)
                    if not self.check_login():
                        raise LoginRequired('职位列表请求需要重新登录。')
                    raise RuntimeError('等待职位列表接口响应超时。')

                body = packet.response.body
                if isinstance(body, (str, bytes)):
                    body = json.loads(body)
                if not isinstance(body, dict):
                    raise RuntimeError('职位列表接口响应格式无效。')
                # 接口报错不能当作空列表成功返回；风控码须停止后续滚动。
                code = str(body.get('code'))
                if code == '36':
                    raise AccessRestricted('职位列表请求触发 code=36，已停止采集。')
                if code != '0':
                    if not self.check_login():
                        raise LoginRequired('职位列表请求需要重新登录。')
                    raise RuntimeError(f'职位列表接口返回错误: code={code}')
                zp_data = body.get('zpData') or {}
                page_jobs = zp_data.get('jobList') or []
                # 多城市搜索结束后浏览器只停在最后一城；保存来源供详情阶段返回。
                page_jobs = [
                    dict(job, _searchUrl=search_url, _searchMaxPages=max_pages)
                    for job in page_jobs
                ]
                has_more = bool(zp_data.get('hasMore'))

                for job in page_jobs:
                    key = self.job_key(job)
                    is_new = key not in jobs_by_key
                    jobs_by_key.setdefault(key, job)
                    if is_new and on_job is not None:
                        # 详情点击会复用同一个页面监听器；先释放列表监听，
                        # 详情结束后再恢复，确保下一页响应不会被详情逻辑消费。
                        page.listen.stop()
                        listener_started = False
                        try:
                            on_job(job)
                        finally:
                            page.listen.start(targets=[job_list_target], is_regex=False)
                            listener_started = True

                page_no = len(pages) + 1
                pages.append({
                    'page': page_no,
                    'hasMore': has_more,
                    'jobCount': len(page_jobs),
                    'jobs': page_jobs,
                })
                logger.info(
                    "职位列表分页完成: page=%s page_jobs=%s unique_jobs=%s has_more=%s",
                    page_no,
                    len(page_jobs),
                    len(jobs_by_key),
                    has_more,
                )

                if not has_more or len(pages) >= max_pages:
                    break

                # hasMore 是列表接口给出的分页信号。不能用整个窗口的 scrollY
                # 判断是否加载成功，因为 BOSS 的结果区可能在内部容器中滚动。
                self.ensure_available(page)
                last = page.ele('css:' + job_card_selector, index=-1, timeout=0)
                if last:
                    last.scroll.to_see()
                    page.actions.move_to(last).scroll(800, 0)
                else:
                    page.actions.scroll(1000000, 0)
                sleep(2)

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
                }, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            logger.info("职位采集完成: jobs=%s save_dir=%s", len(jobs), save_dir)
            self._last_search_url = search_url
            return jobs
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
    def _check_detail_responses(page) -> None:
        """只观察网页自然请求中的风控码；响应缺失不影响 DOM 提取。"""
        for _ in range(20):
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
                    return {
                        jobId, title, description: body.innerText.trim(),
                        bossActiveTime: root.querySelector('.boss-online-tag')?.innerText?.trim() || ''
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


    def get_job_card(self, job: dict, timeout: float = 20) -> dict | None:
        """在后台标签页打开职位详情，读取 DOM 后合并到原摘要。

        保留 encryptJobId/securityId/lid 等下游字段；不伪造网页未提供的
        friendStatus 等字段。主搜索页保持不动；普通超时返回 None，登录
        或风控异常向上抛出并保留问题标签页供人工处理。
        """
        if not job.get('encryptJobId') or not job.get('jobName'):
            return None
        page = self.get_page()
        job_id = quote(str(job['encryptJobId']), safe='')
        detail_url = f'https://www.zhipin.com/job_detail/{job_id}.html'
        detail_page = page.new_tab(background=True)
        blocked = False
        detail_page.listen.start(targets=['/wapi/zpgeek/'], is_regex=False)
        try:
            detail_page.get(detail_url)
            detail_page.wait.load_start()
            deadline = monotonic() + timeout
            previous = None
            while monotonic() < deadline:
                self._check_detail_responses(detail_page)
                self._ensure_detail_available(detail_page)
                detail = self._read_job_detail_dom(detail_page)
                if self._detail_matches_job(detail, job):
                    # 等待连续两次正文一致，减少读取异步更新中间状态的机会。
                    fingerprint = (detail['jobId'], detail['title'], detail['description'])
                    if fingerprint == previous:
                        return {
                            **job,
                            'postDescription': detail['description'],
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
        finally:
            try:
                detail_page.listen.stop()
            except Exception:
                logger.debug('停止详情请求观测失败', exc_info=True)
            # 风控页面保持原样供人工查看；正常详情标签读取后立即关闭。
            if not blocked:
                detail_page.close()


    def get_job_cards(self, jobs: list[dict], interval: tuple[float, float] = (1, 2)) -> list[dict | None]:
        """逐个读取职位详情，保留顺序和间隔；风控异常立即终止批次。

        返回:
            与 jobs 等长的列表，每项是 jobCard 字典或 None（失败的职位）。
            被验证码或访问限制中断时，异常的 partial_cards 保留已处理结果。
        """
        cards = []
        for index, job in enumerate(jobs, 1):
            if index > 1:
                sleep(uniform(*interval))
            try:
                card = self.get_job_card(job)
            except (VerificationRequired, AccessRestricted) as exc:
                # 停止后续点击，同时把已有结果交给状态图保存，避免整批丢失。
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
        """取 bst cookie，投递接口要求放进 Zp_token 头。"""
        for cookie in self.get_page().cookies():
            if cookie.get('name') == 'bst':
                return cookie.get('value', '')
        return ''

    def _get_push_page(self):
        """获取复用的后台投递标签页，不切换用户当前正在看的页面。"""
        if self._push_page is not None:
            try:
                # 访问 url 是轻量的存活检查；标签页被用户或浏览器关闭时会抛异常。
                self._push_page.url
                return self._push_page
            except Exception:
                logger.info("投递后台标签页已失效，将重新创建")
                self._push_page = None

        page = self.get_page()
        self._push_page = page.new_tab(
            'https://www.zhipin.com/',
            background=True,
        )
        self._push_page.wait.load_start()
        return self._push_page

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
    async (url, token) => {
        const resp = await fetch(url, {
            method: 'POST',
            credentials: 'include',
            headers: {'Zp_token': token},
        });
        return await resp.text();
    }
    """

    def push_job(self, job: dict, retries: int = 3) -> tuple[bool, str]:
        """向单个职位发起投递（打招呼）。

        返回:
            (是否成功, 消息)
        """
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
                # run_async_js() 在当前 DrissionPage 版本不返回 Promise 结果；
                # run_js() 会等待 async 函数完成并返回 fetch 的响应文本。
                page = self._get_push_page()
                raw = page.run_js(self._push_js, url, token)
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
        if self._page is not None:
            logger.info(
                "正在关闭浏览器连接: owns_browser=%s",
                self._owns_browser,
            )
            if self._push_page is not None:
                try:
                    self._push_page.close()
                except Exception:
                    logger.debug("关闭投递后台标签页失败", exc_info=True)
                finally:
                    self._push_page = None
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
            # 关闭后复位就绪信号，避免后台线程误判浏览器仍然可用。
            self._ready.clear()
            logger.info("浏览器连接已关闭")


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
