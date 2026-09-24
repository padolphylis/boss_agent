import json
import pathlib
import unicodedata
from dataclasses import dataclass
from threading import Event, RLock
from random import uniform
from typing import Callable
from time import monotonic, sleep
from urllib.request import urlopen
from urllib.parse import parse_qs, urlencode, urlparse

from DrissionPage import ChromiumPage, ChromiumOptions

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


base_dir = pathlib.Path(__file__).resolve().parent
user_data_dir = base_dir.parent / 'user_data' / 'user_data'
save_dir = base_dir / 'jobs_data'
user_data_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)

job_list_target = '/wapi/zpgeek/search/joblist.json'
push_api = 'https://www.zhipin.com/wapi/zpgeek/friend/add.json'
daily_limit_hint = '您今天已与120位BOSS沟通'
max_joblist_pages = 8
max_empty_scrolls = 3
chat_url = 'https://www.zhipin.com/web/geek/chat'

# 职位详情改为通过列表中的正常点击读取 DOM。这里保留多个选择器，
# 以兼容 BOSS 页面不同版本的详情容器；不会直接请求 job/card.json。
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
auth_cookie_names = frozenset({'bst', 'wt2', 'zp_at'})
login_entry_texts = ('登录/注册', '立即登录', '登录账号，查看更多好职位')
login_entry_selector = 'header a, header button, nav a, nav button, .header a, .header button, .header-nav a, .header-nav button'

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
    max_empty_scrolls = 3




class BrowserManager:
    def __init__(self):
        self.user_data_dir = user_data_dir
        self._page:ChromiumPage|None = None
        self._last_search_url = ''
        self._detail_list_exhausted = False
        # 浏览器就绪信号：_page 首次可用时置位，供聊天监听等后台线程等待。
        self._ready = Event()


    def get_page(self) -> ChromiumPage:
        if self._page is None:
            raise ValueError("Browser not started")
        return self._page


    def start(self):
        """启动浏览器；已启动则直接复用，避免重复占用同一 user_data_dir。"""
        if self._page is None:
            logger.info("正在启动浏览器: user_data_dir=%s port=%s", self.user_data_dir, debug_port)
            # DrissionPage 要求 DevTools 下至少存在一个 page。旧实例可能只
            # 占着端口但没有页面，先清理后再启动，避免连接检测卡住 30 秒。
            if not _debug_pages(debug_port):
                _stop_stale_browser(debug_port, self.user_data_dir)
            options = ChromiumOptions()
            options.headless(False)
            options.set_local_port(debug_port)
            options.set_user_data_path(str(self.user_data_dir))
            options.set_argument('--no-startup-window', False)
            options.set_argument('--new-window')
            self._page = ChromiumPage(addr_or_opts=options)
            logger.info("浏览器启动完成")
            self._ready.set()
        else:
            logger.debug("复用已启动的浏览器页面")
        return self


    def is_ready(self) -> bool:
        """浏览器是否已经启动并持有可用页面。"""
        return self._page is not None


    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """等待浏览器就绪。

        返回 True 表示已就绪；超时返回 False。传入 timeout=None 表示一直等待，
        适用于后台监听线程在浏览器尚未启动时保持待命。
        """
        if self._page is not None:
            return True
        return self._ready.wait(timeout)


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
        try:
            self.ensure_available()
        except VerificationRequired:
            self.wait_for_verification()
            self.ensure_available()


    def check_login(self) -> bool:
        """用页面状态、会话 Cookie 和登录入口共同判断登录状态。

        ``window._PAGE.isLogin`` 在页面异步初始化期间可能是 ``None``，
        所以不能把这个单一信号当成未登录；这里只读取 Cookie 名称，
        不读取或记录 Cookie 值。
        """
        page = self.get_page()
        try:
            page_flag = page.run_js(
                """
                const value = window._PAGE && window._PAGE.isLogin;
                return value === true ? true : value === false ? false : null;
                """
            )
        except Exception:
            page_flag = None

        try:
            cookie_names = {
                str(cookie.get('name'))
                for cookie in (page.cookies() or [])
                if cookie.get('name')
            }
        except Exception:
            cookie_names = set()

        if page_flag is True:
            return True
        if page_flag is False and not (cookie_names & auth_cookie_names):
            return False

        # 会话 Cookie 只是辅助信号；过期会话若出现登录入口，应返回未登录。
        # 详情阶段还会检查实际页面是否要求登录，不把这里的推断当成永久有效。
        try:
            text = str(page.run_js("return document.body ? document.body.innerText : '';") or '')
            has_login_entry = self._has_login_entry(page)
        except Exception:
            return False
        return bool(text.strip()) and bool(cookie_names & auth_cookie_names) and not has_login_entry


    def login(self):
        """登录。"""
        while True:
            if self.check_login():
                break
            sleep(1)

        return self.check_login()



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
        # 依据 data/filter_options.json 完成，这里只负责拼接 URL。
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
            # 通过 URL 参数加载筛选条件。
            page.listen.start(targets=[job_list_target], is_regex=False)
            listener_started = True
            page.get(search_url)
            self._detail_list_exhausted = False
            page.wait.load_start()
            has_more = True
            while has_more and len(pages) < max_pages:
                self.ensure_available(page)
                packet = page.listen.wait(timeout=10, raise_err=False)
                if not packet:
                    self.ensure_available(page)
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

    def _job_list_snapshot(self) -> list[dict]:
        """只读取卡片链接中的职位 ID；不按列表下标或模糊标题匹配。"""
        return self.get_page().run_js(r"""
            return [...document.querySelectorAll(arguments[0])].map(card => {
                const link = card.querySelector('a[href*="/job_detail/"]');
                const path = link ? new URL(link.href, location.href).pathname : '';
                const match = path.match(/^\/job_detail\/([^/]+)\.html$/);
                return {id: match ? match[1] : ''};
            }).filter(card => card.id);
        """, job_card_selector) or []

    def _find_job_card_element(self, job_id: str):
        """重新定位元素，避免列表异步排序后点击到另一个职位。"""
        return self.get_page().run_js("""
            const [selector, jobId] = arguments;
            return [...document.querySelectorAll(selector)].find(card => {
                const link = card.querySelector('a[href*="/job_detail/"]');
                return link && new URL(link.href, location.href).pathname
                    === '/job_detail/' + jobId + '.html';
            }) || null;
        """, job_card_selector, job_id)

    def _locate_job_card(self, job: dict, timeout: float):
        """返回来源列表并有限滚动查找目标；消失的职位按失败处理。"""
        page = self.get_page()
        source_url = job.get('_searchUrl') or self._last_search_url or page.url
        parsed = urlparse(source_url)
        if parsed.hostname != 'www.zhipin.com' or parsed.path != '/web/geek/jobs':
            raise ValueError('缺少职位来源搜索页，请先调用 get_job_list()。')
        if page.url != source_url:
            self.ensure_available()
            page.get(source_url)
            self._detail_list_exhausted = False

        deadline = monotonic() + timeout
        previous_ids = set()
        empty_scrolls = 0
        scrolls = 0
        next_scroll = monotonic() + 1
        while monotonic() < deadline:
            self._ensure_detail_available(page)
            element = self._find_job_card_element(str(job['encryptJobId']))
            if element:
                return element
            if self._detail_list_exhausted:
                return None

            ids = {card['id'] for card in self._job_list_snapshot()}
            if ids and monotonic() >= next_scroll:
                empty_scrolls = empty_scrolls + 1 if ids == previous_ids else 0
                if (empty_scrolls >= max_empty_scrolls
                        or scrolls >= job.get('_searchMaxPages', max_joblist_pages)):
                    self._detail_list_exhausted = True
                    return None
                previous_ids = ids
                # 鼠标位于最后一张卡片上，兼容页面滚动和列表自身的滚动容器。
                last = page.ele('css:' + job_card_selector, index=-1, timeout=0)
                if last:
                    last.scroll.to_see()
                    page.actions.move_to(last).scroll(800, 0)
                scrolls += 1
                next_scroll = monotonic() + 2
            sleep(0.25)
        return None

    def _ensure_detail_available(self, page) -> None:
        """详情读取期间立即停止验证、访问限制或登录失效，不自动重试。"""
        self.ensure_available(page)
        if '/web/user/' in (page.url or ''):
            raise AccessRestricted('登录已失效，请人工登录后重试。')
        if self._has_login_entry(page):
            raise AccessRestricted('页面要求登录，请人工登录后重试。')

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
        """按职位 ID 点击列表卡片，读取 DOM 后合并到原摘要。

        保留 encryptJobId/securityId/lid 等下游字段；不伪造网页未提供的
        friendStatus 等字段。普通超时返回 None，登录或风控异常向上抛出。
        """
        if not job.get('encryptJobId') or not job.get('jobName'):
            return None
        page = self.get_page()
        element = self._locate_job_card(job, timeout)
        if not element:
            logger.warning('来源列表中未找到职位: title=%s', job.get('jobName', ''))
            return None

        source_url = page.url
        original_tabs = set(page.tab_ids)
        detail_page = page
        listening_pages = [page]
        blocked = False
        page.listen.start(targets=['/wapi/zpgeek/'], is_regex=False)
        try:
            element.click(by_js=False)
            deadline = monotonic() + timeout
            previous = None
            while monotonic() < deadline:
                self._check_detail_responses(page)
                # 某些版本会新开详情标签，只接管本次点击产生的目标职位标签。
                if detail_page is page:
                    for tab_id in set(page.tab_ids) - original_tabs:
                        tab = page.get_tab(tab_id)
                        target = urlparse(tab.url)
                        is_detail = target.path == f"/job_detail/{job['encryptJobId']}.html"
                        is_block = (
                            target.path in ('/403.html', '/web/passport/zp/verify.html')
                            or target.path.startswith('/web/user/')
                        )
                        if target.hostname == 'www.zhipin.com' and (is_detail or is_block):
                            detail_page = tab
                            detail_page.listen.start(targets=['/wapi/zpgeek/'], is_regex=False)
                            listening_pages.append(detail_page)
                            break
                if detail_page is not page:
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
            logger.warning('点击后详情未匹配或加载超时: title=%s timeout=%s', job['jobName'], timeout)
            return None
        except (VerificationRequired, AccessRestricted):
            blocked = True
            raise
        finally:
            for listening_page in listening_pages:
                try:
                    listening_page.listen.stop()
                except Exception:
                    logger.debug('停止详情请求观测失败', exc_info=True)
            # 风控页面保持原样供人工查看；正常侧栏无需刷新列表。
            if not blocked:
                if detail_page is not page:
                    detail_page.close()
                elif page.url != source_url:
                    page.back()
                    self._detail_list_exhausted = False

    def get_job_cards(self, jobs: list[dict], interval: tuple[float, float] = (1, 2)) -> list[dict | None]:
        """逐个点击并读取职位详情，保留顺序和间隔；风控异常立即终止批次。

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
        page = self.get_page()
        try:
            raw = page.run_async_js(self._push_js, url, token)
            payload = json.loads(raw)
        except Exception as exc:
            return False, f'结果未知：{type(exc).__name__}: {exc}'

        code = payload.get('code')
        message = payload.get('message') or ''
        remind = (
            ((payload.get('zpData') or {}).get('bizData') or {})
            .get('chatRemindDialog') or {}
        ).get('content') or ''

        if code == 0 and message == 'Success':
            return True, 'Success'
        if daily_limit_hint in remind:
            return True, remind
        return False, remind or message or f'code={code}'

    def close(self):
        if self._page is not None:
            logger.info("正在关闭浏览器")
            self._page.quit()
            self._page = None
            # 关闭后复位就绪信号，避免后台线程误判浏览器仍然可用。
            self._ready.clear()
            logger.info("浏览器已关闭")


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
