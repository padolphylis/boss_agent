"""职位详情集成的回归测试：假页面验证流程，本地 HTML 验证真实 DOM 选择器。

运行：python -m unittest discover -s test -p test_browser_job_details.py -v
所有浏览器用例仅打开本地 HTML，不使用账号，也不访问 BOSS 网站。
"""

import json
import os
import pathlib
import tempfile
import unittest
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, quote, urlparse

from DrissionPage import ChromiumOptions, ChromiumPage

from browser import AccessRestricted, BrowserManager, VerificationRequired


SEARCH_URL = 'https://www.zhipin.com/web/geek/jobs?query=AI&city=101210100'
JOB = {
    'encryptJobId': 'job-b', 'jobName': 'AI Agent', 'brandName': 'Example',
    'securityId': 'test-security', 'lid': 'test-lid', '_searchUrl': SEARCH_URL,
}
DETAIL = {
    'jobId': 'job-b',
    'title': 'AI Agent',
    'description': 'Build and maintain AI applications. 登录/注册相关说明仅属于岗位正文。',
}


class DetailFlowTests(unittest.TestCase):
    def setUp(self):
        self.manager = BrowserManager()
        self.page = Mock(url=SEARCH_URL, tab_ids=['list'])
        self.page.listen.wait.return_value = None
        self.manager._page = self.page
        self.element = Mock()
        self.manager._locate_job_card = Mock(return_value=self.element)
        self.manager._ensure_detail_available = Mock()
        self.manager._read_job_detail_dom = Mock(return_value=DETAIL)
        self.sleep = patch('browser.sleep').start()
        self.addCleanup(patch.stopall)

    def test_old_detail_is_rejected_and_summary_fields_are_preserved(self):
        stale = dict(DETAIL, jobId='job-a')
        self.manager._read_job_detail_dom.side_effect = [stale, DETAIL, DETAIL]
        card = self.manager.get_job_card(JOB)
        self.assertEqual(card, dict(JOB, postDescription=DETAIL['description']))
        self.assertEqual(self.manager._read_job_detail_dom.call_count, 3)
        self.element.click.assert_called_once_with(by_js=False)
        self.page.get.assert_not_called()  # 已在列表上时不能导航到详情接口。
        self.page.listen.stop.assert_called_once()
        self.assertNotIn('postDescription', JOB)

    def test_title_prefix_is_not_a_match(self):
        self.assertFalse(self.manager._detail_matches_job(dict(DETAIL, title='AI Agent研发'), JOB))
        self.assertFalse(self.manager._detail_matches_job(dict(DETAIL, jobId='other'), JOB))
        self.assertTrue(self.manager._detail_matches_job(dict(DETAIL, title='AI\nAgent'), JOB))

    def test_timeout_returns_none_and_cleans_listener(self):
        self.manager._read_job_detail_dom.return_value = {}
        with patch('browser.monotonic', side_effect=[0, 1, 3]):
            self.assertIsNone(self.manager.get_job_card(JOB, timeout=2))
        self.page.listen.stop.assert_called_once()

    def test_code_36_stops_batch_without_another_click(self):
        for body in ({'code': 36}, json.dumps({'code': '36'}), b'{"code":36}'):
            with self.subTest(body=body):
                self.page.listen.wait.return_value = SimpleNamespace(response=SimpleNamespace(body=body))
                self.element.reset_mock()
                with self.assertRaises(AccessRestricted):
                    self.manager.get_job_cards([JOB, dict(JOB, encryptJobId='next')], interval=(0, 0))
                self.element.click.assert_called_once()
                self.page.back.assert_not_called()

    def test_captcha_keeps_page_and_stops_batch(self):
        self.manager._ensure_detail_available.side_effect = VerificationRequired('captcha')
        with self.assertRaises(VerificationRequired):
            self.manager.get_job_cards([JOB, JOB])
        self.element.click.assert_called_once()
        self.page.listen.stop.assert_called_once()
        self.page.back.assert_not_called()

    def test_batch_keeps_order_and_failed_slots(self):
        self.manager.get_job_card = Mock(side_effect=[None, dict(JOB, postDescription='description')])
        cards = self.manager.get_job_cards([{'encryptJobId': 'missing'}, JOB], interval=(0, 0))
        self.assertEqual(len(cards), 2)
        self.assertIsNone(cards[0])
        self.assertEqual(cards[1]['encryptJobId'], JOB['encryptJobId'])

    def test_source_city_is_restored_before_lookup(self):
        manager = BrowserManager()
        manager._page = self.page
        self.page.url = SEARCH_URL.replace('101210100', '101010100')
        manager._detail_list_exhausted = True
        manager.ensure_available = Mock()
        manager._ensure_detail_available = Mock()
        manager._find_job_card_element = Mock(return_value=self.element)
        self.assertIs(manager._locate_job_card(JOB, 2), self.element)
        self.page.get.assert_called_once_with(SEARCH_URL)
        manager._find_job_card_element.assert_called_once_with('job-b')
        self.assertFalse(manager._detail_list_exhausted)

    def test_missing_job_has_bounded_scrolling(self):
        manager = BrowserManager()
        manager._page = self.page
        manager._ensure_detail_available = Mock()
        manager._find_job_card_element = Mock(return_value=None)
        manager._job_list_snapshot = Mock(return_value=[{'id': 'other'}])
        with patch('browser.monotonic', side_effect=count(0, 0.5)):
            self.assertIsNone(manager._locate_job_card(JOB, 100))
            scrolls = self.page.actions.move_to.return_value.scroll.call_count
            self.assertGreater(scrolls, 0)
            self.assertLessEqual(scrolls, 8)
            self.assertTrue(manager._detail_list_exhausted)
            self.assertIsNone(manager._locate_job_card(dict(JOB, encryptJobId='missing'), 100))
            self.assertEqual(self.page.actions.move_to.return_value.scroll.call_count, scrolls)

    def test_same_tab_navigation_returns_to_list(self):
        def navigate(**kwargs):
            self.page.url = 'https://www.zhipin.com/job_detail/job-b.html'
        self.element.click.side_effect = navigate
        self.assertIsNotNone(self.manager.get_job_card(JOB))
        self.page.back.assert_called_once()

    def test_new_detail_tab_is_closed_but_list_tab_is_kept(self):
        tab = Mock(url='https://www.zhipin.com/job_detail/job-b.html')
        tab.listen.wait.return_value = None
        self.page.get_tab.return_value = tab
        def open_tab(**kwargs):
            self.page.tab_ids = ['list', 'detail']
        self.element.click.side_effect = open_tab
        self.assertIsNotNone(self.manager.get_job_card(JOB))
        self.manager._read_job_detail_dom.assert_called_with(tab)
        tab.close.assert_called_once()
        tab.listen.stop.assert_called_once()
        self.page.close.assert_not_called()

    def test_verification_in_new_tab_is_kept_for_manual_handling(self):
        tab = Mock(url='https://www.zhipin.com/web/passport/zp/verify.html')
        tab.listen.wait.return_value = None
        self.page.get_tab.return_value = tab
        def open_tab(**kwargs):
            self.page.tab_ids = ['list', 'verify']
        self.element.click.side_effect = open_tab
        self.manager._ensure_detail_available.side_effect = VerificationRequired('captcha')
        with self.assertRaises(VerificationRequired):
            self.manager.get_job_card(JOB)
        tab.close.assert_not_called()
        tab.listen.stop.assert_called_once()

    def test_login_without_page_flag_and_expired_cookie(self):
        self.page.cookies.return_value = [{'name': 'bst'}, {'name': 'wt2'}]
        self.page.run_js.side_effect = [None, '消息 简历 用户', False]
        self.assertTrue(self.manager.check_login())
        self.page.run_js.side_effect = [None, '登录/注册', True]
        self.assertFalse(self.manager.check_login())
        self.page.run_js.side_effect = [None, '', False]
        self.assertFalse(self.manager.check_login())

    def test_job_list_code_36_stops_before_scroll(self):
        manager = BrowserManager()
        manager._page = self.page
        manager.ensure_or_wait = Mock()
        manager.ensure_available = Mock()
        self.page.listen.wait.return_value = SimpleNamespace(
            response=SimpleNamespace(body={'code': 36})
        )
        with self.assertRaises(AccessRestricted):
            manager.get_job_list('AI', city='101210100', max_pages=2)
        self.page.actions.scroll.assert_not_called()
        self.page.listen.stop.assert_called_once()

    def test_job_list_url_uses_boss_filter_parameter_names(self):
        manager = BrowserManager()
        manager._page = self.page
        manager.ensure_or_wait = Mock()
        manager.ensure_available = Mock()
        self.page.listen.wait.return_value = SimpleNamespace(
            response=SimpleNamespace(body={
                "code": "0",
                "zpData": {"jobList": [], "hasMore": False},
            })
        )

        manager.get_job_list(
            "ai应用工程师",
            city="100010000",
            job_type_code=1901,
            salary_code=403,
            experience_code=108,
            degree_code=208,
            scale_code=302,
            stage_code=801,
        )

        search_url = self.page.get.call_args.args[0]
        query = parse_qs(urlparse(search_url).query)
        self.assertEqual(query, {
            "city": ["100010000"],
            "jobType": ["1901"],
            "salary": ["403"],
            "experience": ["108"],
            "degree": ["208"],
            "scale": ["302"],
            "stage": ["801"],
            "query": ["ai应用工程师"],
        })


class DetailDomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 自动端口使用临时配置目录，与人工登录用的浏览器完全隔离。
        cls.page = ChromiumPage(ChromiumOptions().auto_port().headless(True))

    @classmethod
    def tearDownClass(cls):
        cls.page.quit()

    def setUp(self):
        html = '''
            <base href="https://www.zhipin.com/">
            <div class="job-card-wrap"><a class="job-name" href="/job_detail/job-a.html">AI Agent</a></div>
            <div class="job-card-wrap active"><a class="job-name" href="/job_detail/job-b.html">AI Agent</a></div>
            <div class="job-detail-container" style="display:none">
              <div class="job-detail-header"><span class="job-name">Hidden</span></div>
              <div class="job-detail-body"><p class="desc">Hidden description</p></div>
            </div>
            <div class="job-detail-container" id="detail">
              <div class="job-detail-header"><span class="job-name">AI Agent</span></div>
              <div class="job-detail-body">
                <p class="desc">Build and maintain AI applications. 登录/注册相关说明仅属于岗位正文。<span style="display:none">Hidden noise</span></p>
                <span class="boss-online-tag">Online</span>
                <a class="more-job-btn" href="/job_detail/job-b.html">More information</a>
              </div>
              <div class="hot-jobs">Unrelated sales and marketing recommendations</div>
            </div>
        '''
        fd, path = tempfile.mkstemp(suffix='.html')
        os.close(fd)
        pathlib.Path(path).write_text(html, encoding='utf-8')
        self.addCleanup(lambda: pathlib.Path(path).unlink(missing_ok=True))
        self.page.get('file://' + path)
        self.manager = BrowserManager()
        self.manager._page = self.page

    def test_dom_uses_detail_header_and_only_description(self):
        detail = self.manager._read_job_detail_dom(self.page)
        self.assertEqual(detail, dict(DETAIL, bossActiveTime='Online'))
        self.assertTrue(self.manager._detail_matches_job(detail, JOB))

    def test_login_word_in_description_is_not_a_login_entry(self):
        self.assertFalse(self.manager._has_login_entry(self.page))

    def test_reordering_cards_does_not_change_target(self):
        self.page.run_js("document.body.prepend(document.querySelector('.job-card-wrap.active'));")
        target = self.manager._find_job_card_element('job-b')
        self.assertIn('/job_detail/job-b.html', target.ele('css:a').attr('href'))
        self.assertEqual(self.manager._job_list_snapshot(), [{'id': 'job-b'}, {'id': 'job-a'}])
        self.assertIsNone(self.manager._find_job_card_element('missing'))

    def test_active_card_does_not_certify_stale_detail(self):
        self.page.run_js("document.querySelector('#detail .more-job-btn').href='/job_detail/job-a.html';")
        detail = self.manager._read_job_detail_dom(self.page)
        self.assertFalse(self.manager._detail_matches_job(detail, JOB))

    def test_container_text_is_not_used_when_description_is_absent(self):
        self.page.run_js("document.querySelector('#detail .desc').remove();")
        self.assertEqual(self.manager._read_job_detail_dom(self.page), {})


if __name__ == '__main__':
    unittest.main()
