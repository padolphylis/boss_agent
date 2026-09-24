"""通过生产 BrowserManager 验证正常点击获取详情，只读取、不投递。

示例：python test/smoke_browser_job_details.py --query AI --city 101210100
列表中间文件写入临时目录，不覆盖生产 jobs.json；结果仅保存正文预览。
"""

import argparse
import json
import pathlib
import socket
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DrissionPage import ChromiumOptions, ChromiumPage
from browser import AccessRestricted, BrowserManager, VerificationRequired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='AI')
    parser.add_argument('--city', default='101210100')
    parser.add_argument('--max-jobs', type=int, default=3)
    parser.add_argument('--max-pages', type=int, default=1)
    parser.add_argument('--result-file', type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / 'jobs_data/browser_detail_smoke.json')
    args = parser.parse_args()
    if args.max_jobs <= 0 or args.max_pages <= 0:
        parser.error('max-jobs 和 max-pages 必须为正数')

    manager = BrowserManager()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    options = ChromiumOptions().headless(False).set_local_port(port)
    options.set_user_data_path(str(manager.user_data_dir))
    manager._page = ChromiumPage(options)
    result = {'status': 'running', 'requested': args.max_jobs, 'completed': 0, 'results': []}
    blocked = False
    try:
        with tempfile.TemporaryDirectory() as output, patch('browser.save_dir', pathlib.Path(output)):
            jobs = manager.get_job_list(args.query, city=args.city, max_pages=args.max_pages)[:args.max_jobs]
            result['login_detected'] = manager.check_login()
            if not result['login_detected']:
                raise AccessRestricted('无法确认登录，请人工检查浏览器。')
            cards = manager.get_job_cards(jobs)
            for job, card in zip(jobs, cards):
                result['results'].append({
                    'job_id': job['encryptJobId'], 'title': job['jobName'],
                    'matched_id': bool(card and card['encryptJobId'] == job['encryptJobId']),
                    'description_chars': len((card or {}).get('postDescription', '')),
                    'description_preview': (card or {}).get('postDescription', '')[:160],
                })
            result['completed'] = sum(card is not None for card in cards)
            result['status'] = 'passed' if result['completed'] == args.max_jobs else 'failed'
    except (AccessRestricted, VerificationRequired) as exc:
        blocked = True
        result.update(status='blocked', reason=str(exc))
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    finally:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        if not blocked:
            manager.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
