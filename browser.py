import json
import pathlib
from dataclasses import dataclass
from random import uniform
from time import sleep
from urllib.parse import parse_qs, urlencode, urlparse

from DrissionPage import ChromiumPage, ChromiumOptions

from get_value import code_book

base_dir = pathlib.Path(__file__).resolve().parent
user_data_dir = base_dir.parent / 'user_data' / 'user_data'
save_dir = base_dir / 'jobs_data'
user_data_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)

job_list_target = '/wapi/zpgeek/search/joblist.json'
job_card_target = '/wapi/zpgeek/job/card.json'
push_api = 'https://www.zhipin.com/wapi/zpgeek/friend/add.json'
daily_limit_hint = '您今天已与120位BOSS沟通'
max_joblist_pages = 8
max_empty_scrolls = 3

@dataclass
class JobConfig:
    job_list_target = '/wapi/zpgeek/search/joblist.json'
    max_joblist_pages = 8
    max_empty_scrolls = 3




class BrowserManager:
    def __init__(self):
        self.user_data_dir = user_data_dir
        self._page:ChromiumPage|None = None


    def get_page(self) -> ChromiumPage:
        if self._page is None:
            raise ValueError("Browser not started")
        return self._page


    def start(self):
        """启动浏览器；已启动则直接复用，避免重复占用同一 user_data_dir。"""
        if self._page is None:
            options = ChromiumOptions()
            options.set_user_data_path(str(self.user_data_dir))
            self._page = ChromiumPage(addr_or_opts=options)
        return self
    

    def goto(self, url: str):
        result = self.start().get_page().get(url)  
        return result


    def __call__(self, url: str) ->ChromiumPage:
        return self.goto(url)


    def check_yan_cheng_ma(self) -> bool:
        """只检查当前页面是否出现可见的验证码组件。"""
        page = self.get_page()
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
        if '/403.html' in current_url:
            return True

        if bool(page.run_js("""
            const text = document.body ? document.body.innerText : '';
            const restrictedTexts = [
                '访问受限',
                '您的IP存在异常行为',
                '您的 IP 存在异常行为',
                '请勿频繁提交刷新请求'
            ];
            return restrictedTexts.some(item => text.includes(item));
        """)):
            return True


    def check_login(self) -> bool:
        """检查登录状态。"""
        return self.get_page().run_js('return window._PAGE && window._PAGE.isLogin;')


    def login(self):
        """登录。"""
        while True:
            is_login = self.get_page().run_js('return window._PAGE && window._PAGE.isLogin;')
            if is_login:
                break

        """检查登录状态。"""
        return self.get_page().run_js('return window._PAGE && window._PAGE.isLogin;')
    


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
    ) -> list:
        """按关键词和筛选条件获取职位列表。"""
        page = self.get_page()
        pages = []
        jobs_by_key = {}
        empty_scrolls = 0
        city_code = self._resolve_city_code(city)
        filters = {
            'query': query,
            'city': city_code,
            'experience': experience_code,
            'degree': degree_code,
            'scale': scale_code,
            'stage': job_type_code,
            'salary': salary_code,
            'industry': industry,
            'position': position,
        }
        search_url = 'https://www.zhipin.com/web/geek/jobs?' + urlencode(filters)

        try:
            # 通过 URL 参数加载筛选条件。
            page.listen.start(targets=[job_list_target], is_regex=False)
            page.get(search_url)
            page.wait.load_start()
            has_more = True
            while has_more and len(pages) < max_pages:
                packet = page.listen.wait(timeout=10)
                if packet is None:
                    raise RuntimeError('等待职位列表接口响应超时。')

                body = packet.response.body
                if isinstance(body, str):
                    body = json.loads(body)
                zp_data = body.get('zpData') or {}
                page_jobs = zp_data.get('jobList') or []
                has_more = bool(zp_data.get('hasMore'))

                for job in page_jobs:
                    jobs_by_key.setdefault(self.job_key(job), job)

                page_no = len(pages) + 1
                pages.append({
                    'page': page_no,
                    'hasMore': has_more,
                    'jobCount': len(page_jobs),
                    'jobs': page_jobs,
                })
                print(f'第 {page_no} 页：{len(page_jobs)} 条，累计 {len(jobs_by_key)} 条')

                if not has_more:
                    break

                before = self._scroll_state(page)
                page.actions.scroll(1000000, 0)
                sleep(2)
                after = self._scroll_state(page)
                if before == after:
                    empty_scrolls += 1
                    if empty_scrolls >= max_empty_scrolls:
                        print('连续滚动未发生变化，停止采集。')
                        break
                else:
                    empty_scrolls = 0

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
            print(f'采集完成：{len(jobs)} 条职位，已保存到 {save_dir}')
            return jobs
        except Exception as e:
            print(f'采集职位列表失败：{e}')
            raise

            


    def get_job_card(self, job: dict, timeout: float = 10) -> dict | None:
        """获取单个职位的详情卡片。

        参数:
            job: 职位摘要字典，至少包含 securityId 和 lid。
            timeout: 监听响应超时秒数。

        返回:
            jobCard 字典（含 postDescription、friendStatus 等），失败返回 None。
        """
        security_id = job.get('securityId', '')
        lid = job.get('lid', '')
        if not security_id or not lid:
            return None

        page = self.get_page()
        params = {
            'securityId': security_id,
            'lid': lid,
            '_': str(int(__import__('time').time() * 1000)),
        }
        card_url = 'https://www.zhipin.com' + job_card_target + '?' + urlencode(params)

        page.listen.start(targets=[job_card_target], is_regex=False)
        page.get(card_url)
        packet = page.listen.wait(timeout=timeout)
        page.listen.stop()

        if not packet:
            return None

        body = packet.response.body
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='replace')
        if isinstance(body, str):
            body = json.loads(body)

        if body.get('code') != 0:
            return None

        return (body.get('zpData') or {}).get('jobCard') or {}

    def get_job_cards(self, jobs: list[dict], interval: tuple[float, float] = (1, 2)) -> list[dict]:
        """批量获取职位详情卡片，自动控制请求间隔。

        返回:
            与 jobs 等长的列表，每项是 jobCard 字典或 None（失败的职位）。
        """
        cards = []
        for index, job in enumerate(jobs, 1):
            if index > 1:
                sleep(uniform(*interval))

            if self.check_yan_cheng_ma():
                print(f'[{index}/{len(jobs)}] 检测到验证码，停止获取。')
                cards.append(None)
                continue

            card = self.get_job_card(job)
            title = job.get('jobName', '')
            if card:
                desc = card.get('postDescription', '')
                print(f'[{index}/{len(jobs)}] {title}：正文 {len(desc)} 字。')
            else:
                print(f'[{index}/{len(jobs)}] {title}：获取失败。')
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

        token = self._read_bst()
        if not token:
            return False, '未登录（bst cookie 为空）'

        url = f'{push_api}?securityId={security_id}&jobId={job_id}&lid={lid}'
        page = self.get_page()
        last_error = ''

        for _ in range(retries):
            try:
                raw = page.run_async_js(self._push_js, url, token)
                payload = json.loads(raw)
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                sleep(0.8)
                continue

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

        return False, f'重试 {retries} 次仍失败：{last_error}'

    def push_jobs(
        self,
        matched: list[dict],
        interval: tuple[float, float] = (3, 5),
    ) -> list[dict]:
        """批量投递。

        参数:
            matched: match_job_content 输出的列表，每项含 job_card。
            interval: 每次投递间隔秒数范围。

        返回:
            [{"job_card": ..., "success": bool, "message": str}, ...]
        """
        results = []
        for index, item in enumerate(matched, 1):
            card = item.get('job_card') or item
            title = card.get('jobName', '') or card.get('postDescription', '')[:20]

            if index > 1:
                sleep(uniform(*interval))

            if self.check_yan_cheng_ma():
                print(f'[{index}/{len(matched)}] 检测到验证码，停止投递。')
                results.append({"job_card": card, "success": False, "message": "验证码阻断"})
                break

            ok, msg = self.push_job(card)
            tag = "成功" if ok else "失败"
            print(f'[{index}/{len(matched)}] {tag}：{title} -> {msg}')
            results.append({"job_card": card, "success": ok, "message": msg})

        return results

    def close(self):
        if self._page is not None:
            self._page.quit()
            self._page = None
    

    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    
        
