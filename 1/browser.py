import json
from pathlib import Path
from time import sleep


# Boss 直聘职位搜索接口，只监听这个接口避免混入静态资源和埋点请求。

# 最多采集的接口响应页数，以及滚动后没有变化时的最大重试次数。
MAX_JOBLIST_PAGES = 8
MAX_EMPTY_SCROLLS = 3

BASE_DIR = Path(__file__).resolve().parent
# 使用独立的浏览器用户目录，保留登录态和浏览器会话数据。
USER_DIR = BASE_DIR / 'browser_files'
# 职位结果输出目录。
SAVE_DIR = BASE_DIR / 'jobs_data'
USER_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)








def _scroll_state(page):
    """获取页面滚动状态，用于判断滚动是否触发了新数据加载。"""
    return page.run_js('''
        return {
            scrollY: window.scrollY,
            height: document.documentElement.scrollHeight,
            viewport: window.innerHeight
        };
    ''')


def collect_jobs(query='前端开发', max_pages=MAX_JOBLIST_PAGES):
    """搜索职位并通过滚动分页收集职位列表。"""

    # pages 保存每次接口响应，jobs_by_key 保存去重后的职位。
    pages = []
    jobs_by_key = {}
    empty_scrolls = 0

    try:

        search_box = page.ele('@placeholder=搜索职位、公司', timeout=10)
        if not search_box:
            raise RuntimeError('找不到职位搜索框。')

        # 先启动监听，再提交搜索，确保不漏掉首次职位列表响应。
        page.listen.start(targets=[JOB_LIST_TARGET], is_regex=False)
        search_box.click()
        search_box.input(f'{query}\n')

        has_more = True
        while has_more and len(pages) < max_pages:
            # 等待 Boss 返回当前页职位 JSON。
            packet = page.listen.wait(timeout=10)
            if packet is None:
                raise RuntimeError('等待职位列表接口响应超时。')

            body = _response_body(packet)
            zp_data = body.get('zpData') or {}
            page_jobs = zp_data.get('jobList') or []
            has_more = bool(zp_data.get('hasMore'))

            # 以职位加密 ID 为主键，避免接口重复返回时重复保存。
            for job in page_jobs:
                jobs_by_key.setdefault(_job_key(job), job)

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

            # Boss 搜索页通过滚动到底部触发下一页接口请求。
            before = _scroll_state(page)
            page.actions.scroll(1000000, 0)
            sleep(2)
            after = _scroll_state(page)
            if before == after:
                empty_scrolls += 1
                if empty_scrolls >= MAX_EMPTY_SCROLLS:
                    print('连续滚动未发生变化，停止采集。')
                    break
            else:
                empty_scrolls = 0

        # 保存去重后的职位，以及每一页的原始职位列表。
        jobs = list(jobs_by_key.values())
        (SAVE_DIR / 'jobs.json').write_text(
            json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (SAVE_DIR / 'joblist_pages.json').write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (SAVE_DIR / 'joblist_summary.json').write_text(
            json.dumps({
                'query': query,
                'pages': len(pages),
                'unique_total': len(jobs),
                'has_more': has_more,
            }, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'采集完成：{len(jobs)} 条职位，已保存到 {SAVE_DIR}')
        return jobs
    finally:
        page.quit()


if __name__ == '__main__':
    collect_jobs()