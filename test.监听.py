import json
import pathlib
from datetime import datetime

from DrissionPage import ChromiumOptions, ChromiumPage


BASE_DIR = pathlib.Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR.parent / 'user_data' / 'user_data'
SAVE_DIR = BASE_DIR / 'jobs_data'
RECORD_FILE = SAVE_DIR / 'manual_network_selcet.jsonl'

USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# 这些请求头可能包含登录凭证，记录时统一脱敏。
SENSITIVE_HEADERS = {
    'authorization',
    'cookie',
    'set-cookie',
    'zp-token',
    'x-zp-token',
}


def safe_value(value):
    """把请求或响应内容转换成可以写入 JSON 的文本。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def safe_headers(headers):
    """复制请求头，并隐藏登录凭证。"""
    if not headers:
        return {}

    result = {}
    for key, value in headers.items():
        if str(key).lower() in SENSITIVE_HEADERS:
            result[key] = '[REDACTED]'
        else:
            result[key] = safe_value(value)
    return result


def get_attr(obj, *names):
    """兼容不同 DrissionPage 版本的属性命名。"""
    for name in names:
        try:
            value = getattr(obj, name, None)
        except (TypeError, KeyError, AttributeError):
            # 某些响应没有完整元数据，访问 headers 等属性会触发库内部异常。
            continue
        if value is not None:
            return value
    return None


def packet_to_record(packet):
    """提取一条网络请求的关键信息。"""
    request = packet.request
    response = packet.response

    request_body = get_attr(request, 'postData', 'post_data', 'data')
    response_body = get_attr(response, 'body', 'text')
    request_headers = get_attr(request, 'headers')
    response_headers = get_attr(response, 'headers')

    return {
        'captured_at': datetime.now().isoformat(timespec='seconds'),
        'request': {
            'url': safe_value(get_attr(request, 'url')),
            'method': safe_value(get_attr(request, 'method')),
            'params': safe_value(get_attr(request, 'params')),
            'post_data': safe_value(request_body),
            'headers': safe_headers(request_headers),
        },
        'response': {
            'url': safe_value(get_attr(response, 'url')),
            'status': safe_value(get_attr(response, 'status', 'status_code')),
            'headers': safe_headers(response_headers),
            'body': safe_value(response_body),
        },
    }


def print_packet_summary(record):
    """在终端显示便于观察的请求摘要。"""
    request = record['request']
    response = record['response']
    url = request['url'] or response['url'] or ''

    print(
        f"[{response['status']}] {request['method'] or ''} {url}"
    )

    keywords = (
        'friend/add',
        'joblist',
        'resume',
        'upload',
        'apply',
        'chat',
        '403',
        'forbidden',
    )
    if any(keyword in url.lower() for keyword in keywords):
        print('  *** 重点请求，请查看 manual_network_selcet.jsonl ***')


def listen_manual_actions():
    """监听浏览器中的手动操作，按 Ctrl+C 停止。"""
    options = ChromiumOptions()
    options.set_user_data_path(str(USER_DATA_DIR))
    page = ChromiumPage(addr_or_opts=options)

    try:
        page.get('https://www.zhipin.com/')
        page.wait.load_start()

        # 不指定 targets，监听当前页面产生的全部网络请求。
        page.listen.start()
        print('网络监听已启动。')
        print('请在浏览器中手动搜索职位、打开职位并提交简历。')
        print(f'记录文件：{RECORD_FILE}')
        print('完成操作后回到终端按 Ctrl+C 停止监听。')

        with RECORD_FILE.open('a', encoding='utf-8') as output:
            while True:
                packet = page.listen.wait(timeout=1)
                # 超时可能返回 False，没有新数据时继续等待。
                if not packet:
                    continue

                record = packet_to_record(packet)
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
                print_packet_summary(record)

    except KeyboardInterrupt:
        print(f'监听已停止，记录保存在：{RECORD_FILE}')
    finally:
        page.quit()


if __name__ == '__main__':
    listen_manual_actions()
