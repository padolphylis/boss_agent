import re
from typing import Any
from uuid import uuid4
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from analysis_resume import ResumeAnalyzer
from analysis_work_content import match_jobs
from browser import BrowserManager
from config import get as cfg
from get_value import code_book
from state import AgentState, SearchParams



def build_graph(browser: BrowserManager):
    """构建职位搜索状态图。每个节点只负责一个明确的状态转换。"""


    def route_on_error(state: AgentState) -> str:
        """有错误就短路到 END，否则继续。"""
        return "failed" if state.error else "continue"


    def analyze_intent(state: AgentState) -> dict[str, Any]:
        """用 LLM 把用户口语解析成结构化搜索条件。"""
        text = state.user_input.strip()
        model_options = {
            "model": cfg("openai_model", "gpt-4o-mini"),
            "temperature": 0,
        }
        base_url = cfg("openai_base_url")
        if base_url:
            model_options["base_url"] = base_url

        parser = ChatOpenAI(**model_options).with_structured_output(SearchParams)
        error_message = ""

        for attempt in range(3):
            prompt = (
                "请把下面的求职需求解析为 SearchParams。"
                "location 只放用户希望去的城市；"
                "exclude_location 只放用户明确排除的城市。"
                "例如'除了上海都去'应解析为 location=[]、"
                "exclude_location=['上海']。未提到的条件保持空值。\n"
                f"用户需求：{text}"
            )
            if error_message:
                prompt += (
                    "\n上一次解析不合规，请根据下面的错误修正后重新输出，"
                    "不要解释，只返回符合 SearchParams 的结果：\n"
                    f"{error_message}"
                )

            try:
                params = parser.invoke(prompt)
                break
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt == 2:
                    # 不写 state.error：LLM 不可用属于可降级情况，
                    # 交给 route_after_intent 走关键词兜底，而不是短路结束。
                    return {
                        "status": "intent_failed",
                        "result": f"LLM 解析失败，已降级为关键词搜索：{error_message}",
                    }

        return {
            "intent": "job_search",
            "search_params": params,
            "status": "intent_analyzed",
        }


    def keyword_search(state: AgentState) -> dict[str, Any]:
        """LLM 不可用时的降级路径：原句作关键词，只保留能查表识别的城市。"""
        text = state.user_input.strip()
        city = code_book.match_city(text)
        query = text.replace(city, " ") if city else text
        return {
            "intent": "keyword_search",
            "search_params": SearchParams(
                zhi_wei=" ".join(query.split()) or text,
                location=[city] if city else [],
            ),
            "status": "keyword_fallback",
        }

    def analyze_resume(state: AgentState) -> dict[str, Any]:
        """解析简历文件，提取纯文本存入状态。"""
        resume_path = state.resume_path.strip()
        if not resume_path:
            return {"resume_analysis": "", "status": "resume_skipped"}

        try:
            analyzer = ResumeAnalyzer(resume_path)
            result = analyzer.analyze()
            return {"resume_analysis": result.text, "status": "resume_analyzed"}
        except Exception as exc:
            return {
                "resume_analysis": "",
                "error": f"简历解析失败: {type(exc).__name__}: {exc}",
                "status": "resume_error",
                "working": False,
            }

    def init_browser(state: AgentState) -> dict[str, Any]:
        """启动浏览器并打开 Boss 首页。"""
        try:
            browser.goto("https://www.zhipin.com/")
            return {"browser": True, "status": "browser_ready"}
        except Exception as exc:
            return {
                "error": f"浏览器启动失败: {type(exc).__name__}: {exc}",
                "status": "browser_failed",
                "working": False,
            }

    def search_jobs(state: AgentState) -> dict[str, Any]:
        """调用驱动层按意图筛选职位，支持多城市搜索与城市排除。"""
        try:
            p = state.search_params
            excluded = {c.strip().rstrip("市") for c in p.exclude_location}

            jobs_by_key: dict[str, dict] = {}
            # location 为空时 city 传空串，由平台按当前城市搜索。
            for city in p.location or [""]:
                jobs = browser.get_job_list(
                    p.zhi_wei or "",
                    city=city,
                    salary_code=code_book.code_of("salary", p.money or ""),
                    experience_code=code_book.code_of("experience", p.experience or ""),
                    scale_code=code_book.code_of("scale", p.scale or ""),
                    degree_code=code_book.code_of("degree", p.degree or ""),
                    job_type_code=code_book.code_of("job_type", p.job_type or ""),
                )
                for job in jobs:
                    city_name = str(job.get("cityName") or "").strip().rstrip("市")
                    if city_name and city_name in excluded:
                        continue
                    jobs_by_key.setdefault(browser.job_key(job), job)

            jobs = list(jobs_by_key.values())
            return {
                "jobs": jobs,
                "result": f"找到 {len(jobs)} 个职位",
                "status": "jobs_fetched",
            }
        except Exception as exc:
            return {
                "error": f"职位搜索失败: {type(exc).__name__}: {exc}",
                "status": "search_failed",
                "working": False,
            }

    def fetch_job_cards(state: AgentState) -> dict[str, Any]:
        """逐个获取职位详情卡片（正文、活跃状态等）。"""
        jobs = state.jobs
        if not jobs:
            return {"job_cards": [], "status": "no_jobs", "working": False}
        try:
            cards = browser.get_job_cards(jobs)
            ok = sum(1 for c in cards if c is not None)
            return {
                "job_cards": [c for c in cards if c is not None],
                "result": f"获取 {ok}/{len(jobs)} 个职位详情",
                "status": "job_cards_fetched",
            }
        except Exception as exc:
            return {
                "error": f"职位详情获取失败: {type(exc).__name__}: {exc}",
                "status": "cards_failed",
                "working": False,
            }

    def match_job_content(state: AgentState) -> dict[str, Any]:
        """对职位正文与用户需求做向量相似度匹配。"""
        cards = state.job_cards
        if not cards:
            return {"matched_jobs": [], "status": "no_cards", "working": False}
        try:
            descriptions = [
                re.sub(r'<[^>]+>', ' ', c.get("postDescription", "")).strip()
                for c in cards
            ]
            ranked = match_jobs(descriptions, state.user_input, threshold=0.3)
            matched = [
                {"job_card": cards[idx], "score": score}
                for idx, score in ranked
            ]
            return {
                "matched_jobs": matched,
                "result": f"匹配到 {len(matched)}/{len(cards)} 个职位",
                "status": "matched",
            }
        except Exception as exc:
            return {
                "error": f"向量匹配失败: {type(exc).__name__}: {exc}",
                "status": "match_failed",
                "working": False,
            }

    def push_jobs(state: AgentState) -> dict[str, Any]:
        """向匹配度达标的职位投递简历。"""
        matched = state.matched_jobs
        if not matched:
            return {"push_results": [], "status": "no_matched", "working": False}
        try:
            results = browser.push_jobs(matched)
            ok = sum(1 for r in results if r.get("success"))
            return {
                "push_results": results,
                "result": f"投递 {ok}/{len(matched)} 个职位",
                "status": "completed",
                "working": False,
                "error": "",
            }
        except Exception as exc:
            return {
                "error": f"投递失败: {type(exc).__name__}: {exc}",
                "status": "push_failed",
                "working": False,
            }


    def route_after_intent(state: AgentState) -> str:
        """LLM 解析成功走标准搜索，否则走关键词降级。"""
        return "llm" if state.intent == "job_search" else "fallback"


    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("keyword_search", keyword_search)
    graph.add_node("analyze_resume", analyze_resume)
    graph.add_node("init_browser", init_browser)
    graph.add_node("search_jobs", search_jobs)
    graph.add_node("fetch_job_cards", fetch_job_cards)
    graph.add_node("match_job_content", match_job_content)
    graph.add_node("push_jobs", push_jobs)

    # 主流程
    graph.add_edge(START, "analyze_intent")
    graph.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        {"llm": "analyze_resume", "fallback": "keyword_search"},
    )
    graph.add_edge("analyze_resume", "init_browser")
    graph.add_edge("keyword_search", "init_browser")

    # 每个关键节点后检查 error，有错短路到 END
    graph.add_conditional_edges(
        "init_browser", route_on_error, {"continue": "search_jobs", "failed": END}
    )
    graph.add_conditional_edges(
        "search_jobs", route_on_error, {"continue": "fetch_job_cards", "failed": END}
    )
    graph.add_conditional_edges(
        "fetch_job_cards", route_on_error, {"continue": "match_job_content", "failed": END}
    )
    graph.add_conditional_edges(
        "match_job_content", route_on_error, {"continue": "push_jobs", "failed": END}
    )
    graph.add_edge("push_jobs", END)

    return graph.compile()


def run_task(user_input: str, browser: BrowserManager, resume: str = "") -> AgentState:
    """执行一次任务并返回最终状态。"""
    state = AgentState(
        task_id=str(uuid4()),
        user_input=user_input,
        resume_path=resume,
        working=True,
        status="started",
    )
    result = build_graph(browser).invoke(state)
    return AgentState.model_validate(result)
