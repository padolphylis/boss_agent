import json
import pathlib
from dataclasses import dataclass
from time import sleep
from urllib.parse import parse_qs, urlencode, urlparse

from DrissionPage import ChromiumPage, ChromiumOptions

BASE_DIR = pathlib.Path(__file__).resolve().parent
user_data_dir = BASE_DIR.parent / 'user_data' / 'user_data'
SAVE_DIR = BASE_DIR / 'jobs_data'
user_data_dir.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

JOB_LIST_TARGET = '/wapi/zpgeek/search/joblist.json'
MAX_JOBLIST_PAGES = 8
MAX_EMPTY_SCROLLS = 3

# Boss 城市编码。城市名称由驱动层转换，编码来源于 Boss 的城市筛选参数。
CITY_CODES = {
    '北京': '101010100',
    '上海': '101020100',
    '杭州': '101210100',
    '濮阳': '101181300',
}

# 薪资
salary = {
    "不限": "",
    "3K以下": 402,
    "3-5K": 403,
    "5-10K": 404,
    "10-20K": 405,
    "20-50K": 406,
    "50K以上": 407,
}
# 工作经验
experience = {
    "不限": "",
    "在校生": 101,
    "应届生": 102,
    "经验不限": 103,
    "1年以内": 104,
    "1-3年": 105,
    "3-5年": 106,
    "5-10年": 107,
    "10年以上": 108,
}
# 公司规模
scale = {
    "不限": "",
    "0-20人": 301,
    "20-99人": 302,
    "100-499人": 303,
    "500-999人": 304,
    "1000-9999人": 305,
    "10000人以上": 306,
}
# 学历
stage = { 
    "不限": "",
    "初中及以下": 201,
    "中专/中技": 202,
    "高中": 203,
    "大专": 204,
    "本科": 205,
    "硕士": 206,
    "博士": 207,
}
# 求职类型
job_type = {
    "不限": "",
    "全职": 1901,
    "兼职": 1902,
    "实习": 1903,
}
@dataclass
class JobConfig:
    JOB_LIST_TARGET = '/wapi/zpgeek/search/joblist.json'
    MAX_JOBLIST_PAGES = 8
    MAX_EMPTY_SCROLLS = 3




class BrowserManager:
    def __init__(self):
        self.user_data_dir = user_data_dir
        self._page:ChromiumPage|None = None


    def get_page(self) -> ChromiumPage:
        if self._page is None:
            raise ValueError("Browser not started")
        return self._page


    def start(self):
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
    


    def _job_key(self, job):
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
        try:
            return CITY_CODES[city_name]
        except KeyError:
            raise ValueError(f'暂不支持城市“{city_name}”，请补充 CITY_CODES 配置。')

    def get_job_list(
        self,
        query: str,
        max_pages: int = MAX_JOBLIST_PAGES,
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
            # 通过 URL 参数加载筛选条件，与 test.py 的测试方式一致。
            page.listen.start(targets=[JOB_LIST_TARGET], is_regex=False)
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
                    jobs_by_key.setdefault(self._job_key(job), job)

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
                    if empty_scrolls >= MAX_EMPTY_SCROLLS:
                        print('连续滚动未发生变化，停止采集。')
                        break
                else:
                    empty_scrolls = 0

            jobs = list(jobs_by_key.values())
            (SAVE_DIR / 'jobs.json').write_text(
                json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            (SAVE_DIR / 'joblist_pages.json').write_text(
                json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            (SAVE_DIR / 'joblist_summary.json').write_text(
                json.dumps({
                    **filters,
                    'pages': len(pages),
                    'unique_total': len(jobs),
                    'has_more': has_more,
                }, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(f'采集完成：{len(jobs)} 条职位，已保存到 {SAVE_DIR}')
            return jobs
        except Exception as e:
            print(f'采集职位列表失败：{e}')
            raise

            


    def close(self):
        if self._page is not None:
            self._page.quit()
            self._page = None
    

    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    
        
