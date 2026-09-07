import json
import pathlib
import re
from time import sleep
from urllib.parse import parse_qs, urlencode, urlparse

BASE_DIR = pathlib.Path(__file__).resolve().parent
_CITY_CODES_FILE = BASE_DIR / 'screenshots' / 'city_codes_crawled.json'
if _CITY_CODES_FILE.exists():
    _city_data = json.loads(_CITY_CODES_FILE.read_text(encoding='utf-8'))
    CITY_CODES = {
        name: str(code)
        for name, code in _city_data.items()
        if str(code).isdigit() and str(code).startswith('101')
    }
else:
    CITY_CODES = {
        '北京': '101010100',
        '上海': '101020100',
        '杭州': '101210100',
        '濮阳': '101181300',
    }


JOB_LIST_TARGET = '/wapi/zpgeek/search/joblist.json'


BASE_DIR = pathlib.Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR.parent / 'user_data' / 'user_data'
SCREENSHOT_DIR = BASE_DIR / 'screenshots'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


FILTERS = {}


def print_filter_codes():
    """打印筛选项与 Boss 请求中使用的编码。"""
    print('\n筛选编码对照：')
    for filter_name, options in FILTERS.items():
        print(f'[{filter_name}]')
        for label, code in options.items():
            print(f'  {label}: {code!r}')


def save_screenshot(page, name):
    """保存当前页面截图并返回文件路径。"""
    path = SCREENSHOT_DIR / name
    page.get_screenshot(path=str(path), full_page=True)
    print(f'截图已保存：{path}')
    return path


def page_observation(page):
    """读取当前页面的 URL、标题和可见文字，辅助核对筛选结果。"""
    observation = page.run_js('''
        return {
            url: window.location.href,
            title: document.title,
            text: document.body ? document.body.innerText : ''
        };
    ''')

    print('\n当前页面：')
    print(f"URL: {observation.get('url', '')}")
    print(f"标题: {observation.get('title', '')}")
    print('\n页面文字前 3000 个字符：')
    print((observation.get('text') or '')[:3000])
    return observation


def reverse_codes(options):
    """把筛选编码反向转换成中文名称。"""
    return {str(code): label for label, code in options.items() if code != ''}


def parse_joblist_request(packet):
    """解析职位列表请求中的实际筛选参数。"""
    post_data = getattr(packet.request, 'postData', '') or ''
    if not isinstance(post_data, str):
        post_data = str(post_data)

    params = parse_qs(post_data, keep_blank_values=True)
    from browser import experience, salary, scale, stage

    reverse_filters = {
        'salary': reverse_codes(salary),
        'experience': reverse_codes(experience),
        'scale': reverse_codes(scale),
        'degree': reverse_codes(stage),
    }
    result = {}
    for name in ('query', 'city', 'salary', 'experience', 'degree', 'scale', 'industry', 'stage', 'position'):
        values = params.get(name, [''])
        raw_value = values[0]
        codes = [code for code in raw_value.split(',') if code]
        result[name] = {
            'raw': raw_value,
            'labels': [reverse_filters.get(name, {}).get(code, f'未知编码:{code}') for code in codes],
        }
    return result


def wait_joblist_packet(page, timeout=10):
    """等待一条职位列表接口响应。"""
    packet = page.listen.wait(timeout=timeout)
    if not packet:
        raise TimeoutError('等待 joblist.json 响应超时。')
    return packet


def save_filter_result(page, packet, name):
    """保存筛选后的截图、请求参数和页面观察结果。"""
    screenshot_path = save_screenshot(page, name)
    filters = parse_joblist_request(packet)
    observation = page_observation(page)
    result = {
        'request_url': packet.request.url,
        'request_method': packet.request.method,
        'filters': filters,
        'screenshot': str(screenshot_path),
        'page': observation,
    }
    result_path = SCREENSHOT_DIR / f'{pathlib.Path(name).stem}.json'
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'筛选参数已保存：{result_path}')
    print('\n实际筛选参数：')
    for key, value in filters.items():
        print(f"  {key}: {value['raw'] or '不限'} -> {', '.join(value['labels']) or '不限'}")


def build_search_url(query, filters):
    """构造职位搜索页 URL，筛选值使用 Boss 页面请求中的编码。"""
    params = {
        'query': query,
        'city': filters.get('city', ''),
        'experience': filters.get('experience', ''),
        'degree': filters.get('degree', ''),
        'scale': filters.get('scale', ''),
        'stage': filters.get('stage', ''),
        'salary': filters.get('salary', ''),
        'industry': filters.get('industry', ''),
        'position': filters.get('position', ''),
    }
    return 'https://www.zhipin.com/web/geek/jobs?' + urlencode(params)


def resolve_city_code(city):
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


def run_city_test(cities=('北京', '杭州', '上海', '濮阳')):
    """测试城市名称能否转换为 city 编码。"""
    results = []
    for city in cities:
        try:
            code = resolve_city_code(city)
            result = {'city': city, 'city_code': code, 'status': 'success'}
            print(f'{city}: {code}')
        except Exception as exc:
            result = {'city': city, 'status': 'failed', 'error': str(exc)}
            print(f'{city}: 解析失败：{exc}')
        results.append(result)
    (SCREENSHOT_DIR / 'city_code_test_results.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return results


def _collect_city_codes(value, result):
    """递归提取响应 JSON 中的城市名称和编码。"""
    if isinstance(value, dict):
        name = next((value.get(key) for key in ('cityName', 'city_name', 'name') if isinstance(value.get(key), str)), None)
        code = next((value.get(key) for key in ('cityCode', 'city_code', 'code') if value.get(key) is not None), None)
        if name and str(code).isdigit() and len(str(code)) >= 6:
            result[name.strip()] = str(code)
        for child in value.values():
            _collect_city_codes(child, result)
    elif isinstance(value, list):
        for child in value:
            _collect_city_codes(child, result)


def crawl_city_codes():
    """通过浏览器网络响应和页面链接批量爬取城市编码。"""
    from DrissionPage import ChromiumOptions, ChromiumPage

    options = ChromiumOptions()
    options.set_user_data_path(str(USER_DATA_DIR))
    page = ChromiumPage(addr_or_opts=options)
    city_codes = dict(CITY_CODES)

    try:
        page.listen.start()
        page.get('https://www.zhipin.com/')
        page.wait.load_start()
        sleep(3)

        links = page.run_js('''
            return Array.from(document.querySelectorAll('a[href]')).map((element) => ({
                text: (element.innerText || element.textContent || '').trim(),
                href: element.href
            }));
        ''') or []
        for link in links:
            code = parse_qs(urlparse(link.get('href', '')).query).get('city', [''])[0]
            name = link.get('text', '')
            if name and code.isdigit():
                city_codes[name] = code

        while True:
            packet = page.listen.wait(timeout=2)
            if not packet:
                break
            body = getattr(packet.response, 'body', None)
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    body = None
            _collect_city_codes(body, city_codes)

        result = dict(sorted(city_codes.items()))
        (SCREENSHOT_DIR / 'city_codes_crawled.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'共爬取城市编码：{len(result)} 个')
        return result
    finally:
        page.quit()


def run_one_test(page, query, filter_name, label, code, index):
    """执行一次单筛选项测试并保存截图和请求记录。"""
    request_filter_name = {
        '薪资': 'salary',
        '工作经验': 'experience',
        '公司规模': 'scale',
        '学历': 'degree',
    }[filter_name]
    filters = {name: '' for name in ('salary', 'experience', 'degree', 'scale')}
    filters[request_filter_name] = str(code)
    url = build_search_url(query, filters)

    page.listen.start(targets=[JOB_LIST_TARGET], is_regex=False)
    page.get(url)
    page.wait.load_start()
    packet = wait_joblist_packet(page)
    sleep(1)

    safe_label = ''.join(char if char.isalnum() else '_' for char in label)
    prefix = f'{index:02d}_{filter_name}_{safe_label}_{code}'
    screenshot_path = save_screenshot(page, f'{prefix}.png')
    request_filters = parse_joblist_request(packet)
    result = {
        'expected': {
            'filter': filter_name,
            'label': label,
            'code': str(code),
        },
        'request_url': packet.request.url,
        'request_method': packet.request.method,
        'request_filters': request_filters,
        'page_url': page.url,
        'screenshot': str(screenshot_path),
    }
    (SCREENSHOT_DIR / f'{prefix}.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    actual = request_filters[request_filter_name]['raw']
    print(f'{prefix}: 期望={code}，实际={actual or "空"}')


def run_test(query='ai应用工程师'):
    """自动逐项测试筛选编码，并保存每次结果。"""
    from DrissionPage import ChromiumOptions, ChromiumPage
    from browser import experience, salary, scale, stage

    filters = {
        '薪资': salary,
        '工作经验': experience,
        '公司规模': scale,
        '学历': stage,
    }
    options = ChromiumOptions()
    options.set_user_data_path(str(USER_DATA_DIR))
    page = ChromiumPage(addr_or_opts=options)
    all_results = []

    try:
        page.get('https://www.zhipin.com/')
        page.wait.load_start()
        print_filter_codes()

        index = 1
        for filter_name, options_map in filters.items():
            for label, code in options_map.items():
                if code == '':
                    continue
                try:
                    run_one_test(page, query, filter_name, label, code, index)
                    all_results.append({
                        'filter': filter_name,
                        'label': label,
                        'code': str(code),
                        'status': 'success',
                    })
                except Exception as exc:
                    print(f'{filter_name}/{label}/{code} 测试失败：{exc}')
                    all_results.append({
                        'filter': filter_name,
                        'label': label,
                        'code': str(code),
                        'status': 'failed',
                        'error': str(exc),
                    })
                index += 1

        (SCREENSHOT_DIR / 'all_filter_test_results.json').write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'测试完成，共 {len(all_results)} 项。')
    finally:
        page.quit()


if __name__ == '__main__':
    crawl_city_codes()