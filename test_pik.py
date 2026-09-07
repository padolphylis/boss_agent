import json
import pathlib
from time import monotonic, sleep

from DrissionPage import ChromiumPage, ChromiumOptions


# 职位列表接口，只监听职位搜索请求。
JOB_LIST_TARGET = '/wapi/zpgeek/search/joblist.json'
MAX_JOBLIST_PAGES = 8
MAX_EMPTY_SCROLLS = 3

BASE_DIR = pathlib.Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR.parent / 'user_data' / 'user_data'
SAVE_DIR = BASE_DIR / 'jobs_data'
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class VerificationRequired(Exception):
    """需要人工处理验证码。"""


class AccessRestricted(Exception):
    """页面访问受限，需要停止抓取。"""


class BrowserManager:
    """封装浏览器启动、页面访问和风控状态检查。"""

    def __init__(self, user_data_dir=USER_DATA_DIR):
        self.user_data_dir = pathlib.Path(user_data_dir)
        self._page: ChromiumPage | None = None

    def start(self):
        """启动浏览器并复用已有用户目录。"""
        if self._page is not None:
            return self

        options = ChromiumOptions()
        options.set_user_data_path(str(self.user_data_dir))
        self._page = ChromiumPage(addr_or_opts=options)
        return self

    def get_page(self) -> ChromiumPage:
        """获取当前浏览器页面。"""
        if self._page is None:
            raise ValueError('Browser not started')
        return self._page

    def goto(self, url: str):
        """打开页面。"""
        self.start()
        page = self.get_page()
        page.get(url)
        page.wait.load_start()
        return page

    def check_yan_cheng_ma(self) -> bool:
        """检查当前页面是否出现验证码组件。"""
        page = self.get_page()
        if bool(page.run_js("""
            const selectors = [
                '.geetest_panel',
                '.geetest_box',
                '.geetest_popup',
                '.nc-container',
                '[class*="captcha"]',
                '[class*="verify"]',
                '[class*="slide"]'
            ];

            return selectors.some((selector) => {
                const element = document.querySelector(selector);
                if (!element) {
                    return false;
                }

                const style = window.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && style.opacity !== '0'
                    && rect.width > 0
                    && rect.height > 0;
            });
        """)):
            return True

        return False

    def check_access_restricted(self) -> bool:
        """检查 403 页面和访问受限提示。"""
        page = self.get_page()
        current_url = page.url or ''
        if '/403.html' in current_url:
            return True

        return bool(page.run_js("""
            const text = document.body ? document.body.innerText : '';
            const restrictedTexts = [
                '访问受限',
                '您的IP存在异常行为',
                '您的 IP 存在异常行为',
                '请勿频繁提交刷新请求'
            ];
            return restrictedTexts.some(item => text.includes(item));
        """))

    def ensure_available(self):
        """检查页面状态，阻止在验证码或受限页面继续抓取。"""
        if self.check_access_restricted():
            raise AccessRestricted('Boss 直聘当前访问受限，停止抓取。')
        if self.check_yan_cheng_ma():
            raise VerificationRequired('检测到验证码，需要人工处理。')

    def wait_for_verification(self, timeout=300):
        """暂停抓取，等待人工完成验证码。"""
        deadline = monotonic() + timeout
        print('检测到验证码，请在浏览器中完成验证。')

        while monotonic() < deadline:
            if self.check_access_restricted():
                raise AccessRestricted('验证过程中页面进入访问受限状态。')
            if not self.check_yan_cheng_ma():
                print('验证码已处理，继续抓取。')
                return
            sleep(1)

        raise TimeoutError('等待验证码处理超时。')

    def close(self):
        """关闭浏览器。"""
        if self._page is not None:
            self._page.quit()
            self._page = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def response_body(packet):
    """读取职位接口响应正文。"""
    body = packet.response.body
    if isinstance(body, bytes):
        body = body.decode('utf-8', errors='ignore')
    if isinstance(body, str):
        return json.loads(body)
    return body


def job_key(job):
    """生成职位去重键。"""
    return job.get('encryptJobId') or (
        f"{job.get('jobName', '')}:"
        f"{job.get('brandName', '')}:"
        f"{job.get('lid', '')}"
    )


def scroll_state(page):
    """记录页面滚动状态。"""
    return page.run_js('''
        return {
            scrollY: window.scrollY,
            height: document.documentElement.scrollHeight,
            viewport: window.innerHeight
        };
    ''')


def save_results(query, pages, jobs, has_more):
    """保存去重后的职位、分页数据和采集摘要。"""
    (SAVE_DIR / 'test_pik_jobs.json').write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (SAVE_DIR / 'test_pik_pages.json').write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (SAVE_DIR / 'test_pik_summary.json').write_text(
        json.dumps({
            'query': query,
            'pages': len(pages),
            'unique_total': len(jobs),
            'has_more': has_more,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def collect_jobs(query='前端开发', max_pages=MAX_JOBLIST_PAGES):
    """搜索并抓取职位列表。"""
    browser = BrowserManager()
    pages = []
    jobs_by_key = {}
    empty_scrolls = 0
    has_more = True

    try:
        browser.goto('https://www.zhipin.com/')
        page = browser.get_page()

        if browser.check_access_restricted():
            raise AccessRestricted('首页访问受限，停止抓取。')

        if not page.run_js('return window._PAGE && window._PAGE.isLogin;'):
            print('当前未登录，请在浏览器中完成登录。')
            sleep(20)
            if not page.run_js('return window._PAGE && window._PAGE.isLogin;'):
                raise RuntimeError('Boss 直聘未登录，无法抓取职位。')

        search_box = page.ele('@placeholder=搜索职位、公司', timeout=10)
        if not search_box:
            raise RuntimeError('找不到职位搜索框。')

        # 先监听，再提交搜索，避免漏掉第一页响应。
        page.listen.start(targets=[JOB_LIST_TARGET], is_regex=False)
        search_box.click()
        search_box.input(f'{query}\n')

        while has_more and len(pages) < max_pages:
            try:
                packet = page.listen.wait(timeout=10)
            except Exception as exc:
                if browser.check_yan_cheng_ma():
                    browser.wait_for_verification()
                    continue
                raise exc

            if packet is None:
                raise TimeoutError('等待职位列表接口响应超时。')

            if browser.check_access_restricted():
                raise AccessRestricted('职位接口请求后页面访问受限。')

            body = response_body(packet)
            zp_data = body.get('zpData') or {}
            page_jobs = zp_data.get('jobList') or []
            has_more = bool(zp_data.get('hasMore'))

            for job in page_jobs:
                jobs_by_key.setdefault(job_key(job), job)

            page_no = len(pages) + 1
            pages.append({
                'page': page_no,
                'hasMore': has_more,
                'jobCount': len(page_jobs),
                'jobs': page_jobs,
            })
            save_results(query, pages, list(jobs_by_key.values()), has_more)
            print(f'第 {page_no} 页：{len(page_jobs)} 条，累计 {len(jobs_by_key)} 条')

            if not has_more:
                break

            before = scroll_state(page)
            page.actions.scroll(1000000, 0)
            sleep(2)

            if browser.check_access_restricted():
                raise AccessRestricted('滚动后页面访问受限。')
            if browser.check_yan_cheng_ma():
                browser.wait_for_verification()

            after = scroll_state(page)
            if before == after:
                empty_scrolls += 1
                if empty_scrolls >= MAX_EMPTY_SCROLLS:
                    print('连续滚动未发生变化，停止抓取。')
                    break
            else:
                empty_scrolls = 0

        jobs = list(jobs_by_key.values())
        save_results(query, pages, jobs, has_more)
        print(f'抓取完成：{len(jobs)} 个唯一职位。')
        return jobs
    finally:
        browser.close()


if __name__ == '__main__':
    collect_jobs()
