"""意图解析回归：所有模型、向量及浏览器调用均使用替身，不访问招聘网站。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from body import _normalize_search_params, _parse_intent_response, build_graph, run_task
from browser import AccessRestricted, BrowserManager
from delivery_store import DeliveryStore
from state import AgentState, SearchParams


REQUEST = "全国都可以,薪资在7以上,包括7,想干ai应用工程师或fde,不想去做漫剧相关的"
PARAMS = {
    "zhi_wei": "AI应用工程师或FDE",
    "money": "7K以上（含7K）",
    "location": [],
    "location_unlimited": True,
    "exclude_keywords": ["漫剧"],
}


def response_for(params):
    """构造 include_raw 的返回值，故意不依赖 SDK 提供的 parsed 字段。"""
    return {
        "raw": AIMessage(content=json.dumps(params, ensure_ascii=False)),
        "parsed": None,
        "parsing_error": None,
    }


class ResponseTests(unittest.TestCase):
    def test_text_fences_and_wrappers(self):
        for params in (PARAMS, {"SearchParams": PARAMS}, {"search_params": PARAMS}):
            for fence in ("", "```json", "```JSON\n", "```\n"):
                with self.subTest(params=params, fence=fence):
                    content = json.dumps(params, ensure_ascii=False)
                    if fence:
                        content = f"{fence}{content}\n```"
                    result = _parse_intent_response({
                        "raw": AIMessage(content=content),
                        "parsed": SearchParams(),
                        "parsing_error": ValueError("SDK parsing failed"),
                    })
                    self.assertEqual(result.zhi_wei, PARAMS["zhi_wei"])
                    self.assertTrue(result.location_unlimited)
                    self.assertEqual(result.money, PARAMS["money"])
                    self.assertEqual(result.exclude_keywords, ["漫剧"])

    def test_tool_arguments_are_validated_before_sdk_defaults(self):
        arguments = "```json" + json.dumps({"SearchParams": PARAMS}) + "```"
        raw = AIMessage(content="", additional_kwargs={"tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "SearchParams", "arguments": arguments},
        }]})
        result = _parse_intent_response({"raw": raw, "parsed": SearchParams()})
        self.assertEqual(result.zhi_wei, PARAMS["zhi_wei"])

    def test_normalized_tool_call(self):
        raw = AIMessage(content="", tool_calls=[{
            "name": "SearchParams", "args": PARAMS, "id": "call-1",
        }])
        self.assertEqual(_parse_intent_response({"raw": raw}).exclude_keywords, ["漫剧"])

    def test_text_content_blocks(self):
        content = json.dumps(PARAMS)
        raw = AIMessage(content=[
            {"type": "text", "text": content[:20]},
            {"type": "text", "text": content[20:]},
        ])
        self.assertEqual(_parse_intent_response(raw).zhi_wei, PARAMS["zhi_wei"])

    def test_invalid_output_never_becomes_empty_search_params(self):
        invalid = [
            "", "not JSON", "```json{}", "{}", "null", "[]", '{"zhi_wei":',
            '{"job_title": "AI"}', '{"zhi_wei": "AI", "unknown": true}',
            '{"SearchParams": {}}', '{"location": "杭州"}',
            '{"zhi_wei": "AI", "location": ["漫剧"]}',
            '{"zhi_wei": "漫剧工程师", "exclude_keywords": ["漫剧"]}',
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                _parse_intent_response({"raw": AIMessage(content=content)})

    def test_refusal_wrong_tool_and_multiple_calls_fail(self):
        wrong_tool = {"name": "Other", "args": PARAMS, "id": "call-1"}
        correct_tool = dict(wrong_tool, name="SearchParams")
        responses = [
            {"raw": None, "parsed": SearchParams()},
            {"raw": AIMessage(content=json.dumps(PARAMS), additional_kwargs={"refusal": "No"})},
            {"raw": AIMessage(content="", tool_calls=[wrong_tool])},
            {"raw": AIMessage(content="", tool_calls=[correct_tool, correct_tool])},
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                _parse_intent_response(response)

    def test_normalization_trims_and_deduplicates(self):
        params = _normalize_search_params({
            "zhi_wei": " AI工程师 ", "location": [" 杭州 ", "杭州"],
            "exclude_keywords": ["", "  ", " 漫剧 ", "漫剧"],
        })
        self.assertEqual(params.zhi_wei, "AI工程师")
        self.assertEqual(params.location, ["杭州"])
        self.assertEqual(params.exclude_keywords, ["漫剧"])

    def test_nationwide_location_alias(self):
        params = _normalize_search_params({"zhi_wei": "AI", "location": ["全国"]})
        self.assertEqual(params.location, [])
        self.assertTrue(params.location_unlimited)

    def test_city_alias_cannot_bypass_exclusion(self):
        with self.assertRaises(ValueError):
            _normalize_search_params({"location": ["上海市"], "exclude_location": ["上海"]})


class IntentFlowTests(unittest.TestCase):
    def _emit_jobs(self, *args, **kwargs):
        jobs = list(self.browser.get_job_list.return_value or [])
        callback = kwargs.get("on_job")
        if callback:
            for job in jobs:
                callback(job)
        return jobs

    def setUp(self):
        self.delivery_dir = tempfile.TemporaryDirectory()
        self.delivery_store = DeliveryStore(Path(self.delivery_dir.name) / "deliveries.db")
        self.delivery_patch = patch("job.DeliveryStore", return_value=self.delivery_store)
        self.delivery_patch.start()
        self.addCleanup(self.delivery_patch.stop)
        self.addCleanup(self.delivery_dir.cleanup)
        self.addCleanup(self.delivery_store.close)
        self.browser = Mock(spec=BrowserManager)
        self.browser.check_login.return_value = True
        self.browser.check_yan_cheng_ma.return_value = False
        self.browser.get_job_list.return_value = []
        self.browser.get_job_list.side_effect = self._emit_jobs
        self.browser.get_job_card.return_value = None
        self.browser.job_key.side_effect = lambda job: job["encryptJobId"]
        self.client = Mock()
        self.parser = self.client.with_structured_output.return_value
        self.parser.invoke.return_value = response_for(PARAMS)
        self.client_patch = patch("body.ChatOpenAI", return_value=self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.config_patch = patch("body.cfg", side_effect=lambda key, default="": {
            "chat_openai_model": "test-model", "chat_openai_api_key": "test-key",
        }.get(key, default))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        # 流程测试验证的是原有匹配入口；Qdrant 的真实行为由独立测试覆盖。
        from vector_store import vector_store

        vector_store.enabled_override = False
        self.addCleanup(lambda: setattr(vector_store, "enabled_override", None))

    def test_screenshot_request_uses_nationwide_search(self):
        state = run_task(REQUEST, self.browser)
        self.assertEqual(state.search_params.zhi_wei, PARAMS["zhi_wei"])
        self.assertNotEqual(state.status, "need_input")
        self.assertEqual(self.browser.get_job_list.call_args.kwargs["city"], "全国")
        self.client.with_structured_output.assert_called_once_with(
            SearchParams, method="function_calling", include_raw=True,
        )

    def test_intent_filter_name_is_transcoded_by_filter_options(self):
        self.parser.invoke.return_value = response_for({
            "zhi_wei": "AI应用工程师",
            "experience": "应届生",
            "location_unlimited": True,
        })
        state = run_task("全国都可以，我是应届生，找 AI 应用工程师", self.browser)

        self.assertNotEqual(state.status, "need_input")
        self.assertEqual(
            self.browser.get_job_list.call_args.kwargs["experience_code"],
            "102",
        )

    def test_multi_select_filters_are_transcoded_and_joined(self):
        self.parser.invoke.return_value = response_for({
            "zhi_wei": "AI应用工程师",
            "experience": ["10年以上", "应届生"],
            "degree": ["博士", "硕士"],
            "scale": ["20-99人", "500-999人", "100-499人"],
            "location_unlimited": True,
        })
        state = run_task("全国都可以，经验和学历、公司规模可多选", self.browser)

        self.assertNotEqual(state.status, "need_input")
        kwargs = self.browser.get_job_list.call_args.kwargs
        self.assertEqual(kwargs["experience_code"], "108,102")
        self.assertEqual(kwargs["degree_code"], "207,206")
        self.assertEqual(kwargs["scale_code"], "302,304,303")

    def test_missing_fields_ask_only_for_missing_information(self):
        examples = [
            ({"zhi_wei": "AI"}, "请补充工作地点"),
            ({"zhi_wei": None, "location_unlimited": True}, "请补充职位"),
            ({"zhi_wei": " ", "location": []}, "请补充职位和工作地点"),
        ]
        for params, question in examples:
            with self.subTest(params=params):
                self.parser.invoke.return_value = response_for(params)
                state = run_task("求职", self.browser)
                self.assertEqual(state.status, "need_input")
                self.assertTrue(state.pending_question.startswith(question))
                self.browser.goto.assert_not_called()

    def test_no_position_is_inferred_from_negative_only_request(self):
        self.parser.invoke.return_value = response_for({
            "zhi_wei": None, "location_unlimited": True, "exclude_keywords": ["销售"],
        })
        state = run_task("全国都可以,不想做销售", self.browser)
        self.assertEqual(state.status, "need_input")
        self.assertIsNone(state.search_params.zhi_wei)
        self.browser.goto.assert_not_called()

    def test_text_retry_when_tools_are_unsupported(self):
        self.parser.invoke.side_effect = RuntimeError("tools unsupported")
        self.client.invoke.return_value = response_for(PARAMS)["raw"]
        state = run_task(REQUEST, self.browser)
        self.assertFalse(state.error)
        self.client.invoke.assert_called_once()
        messages = self.client.invoke.call_args.args[0]
        self.assertEqual(messages[1], {"role": "user", "content": REQUEST})
        self.assertIn('"exclude_keywords"', messages[0]["content"])

    def test_invalid_response_is_retried_instead_of_asking_user(self):
        self.parser.invoke.return_value = response_for({"job_title": "AI"})
        self.client.invoke.return_value = response_for(PARAMS)["raw"]
        state = run_task(REQUEST, self.browser)
        self.assertNotEqual(state.status, "need_input")
        self.client.invoke.assert_called_once()

    def test_repeated_failure_stops_before_browser_and_hides_raw_content(self):
        secret = "private-response-body"
        self.parser.invoke.side_effect = RuntimeError(secret)
        self.client.invoke.side_effect = RuntimeError(secret)
        with self.assertLogs("body", level="DEBUG") as logs:
            state = run_task(REQUEST, self.browser)
        self.assertEqual(state.status, "intent_failed")
        self.assertFalse(state.working)
        self.assertTrue(state.error)
        self.assertEqual(self.client.invoke.call_count, 2)
        self.assertNotIn(secret, str(logs.output))
        self.assertNotIn(REQUEST, str(logs.output))
        self.browser.goto.assert_not_called()

    def test_client_setup_failure_is_reported_as_intent_failure(self):
        with patch("body.ChatOpenAI", side_effect=ValueError("invalid client")):
            state = run_task(REQUEST, self.browser)
        self.assertEqual(state.status, "intent_failed")
        self.browser.goto.assert_not_called()

    def test_city_exclusions_remain_effective_with_nationwide_search(self):
        self.parser.invoke.return_value = response_for(dict(PARAMS, exclude_location=["上海"]))
        self.browser.get_job_list.return_value = [
            {"encryptJobId": "excluded", "cityName": "上海市"},
            {"encryptJobId": "allowed", "cityName": "杭州"},
        ]
        self.browser.get_job_card.return_value = None
        state = run_task(REQUEST, self.browser)
        self.assertEqual([job["encryptJobId"] for job in state.jobs], ["allowed"])

    def test_excluded_jobs_never_reach_embedding_or_push(self):
        cards = [
            {"encryptJobId": "title", "jobName": "漫剧AI", "postDescription": "AI"},
            {"encryptJobId": "body", "jobName": "AI", "postDescription": "制作漫剧"},
            {"encryptJobId": "industry", "brandIndustry": "漫剧", "postDescription": "AI"},
            {"encryptJobId": "allowed", "jobName": "AI", "postDescription": "<p>RAG应用</p>"},
        ]
        self.browser.get_job_list.return_value = cards
        self.browser.get_job_card.side_effect = cards
        self.browser.push_job.return_value = (True, "Success")
        with patch("matcher.match_jobs", return_value=[(0, 0.9)]) as match_jobs:
            state = run_task(REQUEST, self.browser)
        match_jobs.assert_called_once_with(["RAG应用"], REQUEST, threshold=0.3)
        self.browser.push_job.assert_called_once_with(cards[-1])
        self.assertEqual(state.push_results[0]["job_card"]["encryptJobId"], "allowed")
        self.assertEqual(state.status, "completed")

    def test_all_excluded_skips_embeddings_and_push(self):
        cards = [{"encryptJobId": "excluded", "postDescription": "漫剧开发"}]
        self.browser.get_job_list.return_value = cards
        self.browser.get_job_card.side_effect = cards
        with patch("matcher.match_jobs") as match_jobs:
            state = run_task(REQUEST, self.browser)
        self.assertEqual(state.status, "completed")
        match_jobs.assert_not_called()
        self.browser.push_job.assert_not_called()

    def test_null_description_and_case_insensitive_exclusion(self):
        graph = build_graph(self.browser)
        state = AgentState(search_params=SearchParams(exclude_keywords=["AIGC"]), job_cards=[
            {"jobName": "aigc", "postDescription": None},
        ])
        with patch("matcher.match_jobs") as match_jobs:
            result = graph.nodes["match_job_content"].invoke(state)
        self.assertEqual(result["matched_jobs"], [])
        match_jobs.assert_not_called()

    def test_jobs_are_matched_in_batches_before_serial_push(self):
        jobs = [
            {"encryptJobId": "one", "jobName": "AI一", "postDescription": "详情一"},
            {"encryptJobId": "two", "jobName": "AI二", "postDescription": "详情二"},
        ]
        self.browser.get_job_list.return_value = jobs
        self.browser.get_job_card.side_effect = jobs
        events = []
        self.browser.push_job.side_effect = lambda job: (
            events.append(f"push:{job['encryptJobId']}") or (True, "Success")
        )
        self.browser.get_job_card.side_effect = lambda job: (
            events.append(f"detail:{job['encryptJobId']}") or job
        )
        # 模拟批次内的所有匹配结果，并确认抓取结束后才操作浏览器投递。
        with patch("matcher.match_jobs", side_effect=lambda descriptions, *_args, **_kwargs: (
            events.append(f"match:{','.join(descriptions)}") or [(0, 0.9), (1, 0.8)]
        )) as match_jobs, patch("job.time.sleep"):
            state = run_task(REQUEST, self.browser)

        match_jobs.assert_called_once_with(["详情一", "详情二"], REQUEST, threshold=0.3)
        self.assertEqual(
            events,
            [
                "detail:one", "detail:two", "match:详情一,详情二",
                "push:one", "push:two",
            ],
        )
        self.assertEqual(self.browser.push_job.call_args_list[0].args, (jobs[0],))
        self.assertEqual(self.browser.push_job.call_args_list[1].args, (jobs[1],))
        self.assertEqual(len(state.push_results), 2)

    def test_push_failure_preserves_previous_results(self):
        jobs = [
            {"encryptJobId": "one", "postDescription": "详情一"},
            {"encryptJobId": "two", "postDescription": "详情二"},
        ]
        self.browser.get_job_list.return_value = jobs
        self.browser.get_job_card.side_effect = jobs
        self.browser.push_job.side_effect = [(True, "Success"), ValueError("push failed")]
        with patch("matcher.match_jobs", return_value=[(0, 0.9), (1, 0.8)]), patch("job.time.sleep"):
            state = run_task(REQUEST, self.browser)

        self.assertEqual(state.status, "push_failed")
        self.assertIn("ValueError: push failed", state.error)
        self.assertEqual([item["encryptJobId"] for item in state.job_cards], ["one", "two"])
        self.assertEqual([item["job_card"]["encryptJobId"] for item in state.matched_jobs], ["one", "two"])
        self.assertEqual([item["job_card"]["encryptJobId"] for item in state.push_results], ["one"])

    def test_interrupted_detail_fetch_preserves_partial_results(self):
        partial = [{"encryptJobId": "done", "postDescription": "already read"}]
        blocked = AccessRestricted("code=36")
        blocked.partial_cards = partial
        self.browser.get_job_list.side_effect = lambda *args, **kwargs: (
            kwargs["on_job"]({"encryptJobId": "done", "jobName": "已读取"}),
            kwargs["on_job"]({"encryptJobId": "blocked", "jobName": "被阻断"}),
            [],
        )[-1]
        self.browser.get_job_card.side_effect = [partial[0], blocked]
        graph = build_graph(self.browser)
        state = AgentState(
            search_params=SearchParams(zhi_wei="AI", location_unlimited=True),
            status="login_verified",
        )
        result = graph.nodes["search_jobs"].invoke(state)
        self.assertEqual(result["job_cards"], partial)
        self.assertEqual(result["status"], "cards_failed")
        self.assertIn("保留 1 个", result["result"])
        self.assertTrue(result["error"])


if __name__ == "__main__":
    unittest.main()
