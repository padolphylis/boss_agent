import re
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from browser import BrowserManager
from state import AgentState


# 常见筛选词映射。城市名称由 browser.py 中的城市编码表负责转换。
SALARY_CODES = {
    "3K以下": 402,
    "3-5K": 403,
    "5-10K": 404,
    "10-20K": 405,
    "20-50K": 406,
    "50K以上": 407,
}
EXPERIENCE_CODES = {
    "在校生": 101,
    "应届生": 102,
    "经验不限": 103,
    "1年以内": 104,
    "1-3年": 105,
    "3-5年": 106,
    "5-10年": 107,
    "10年以上": 108,
}
DEGREE_CODES = {
    "初中及以下": 201,
    "中专/中技": 202,
    "高中": 203,
    "大专": 204,
    "本科": 205,
    "硕士": 206,
    "博士": 207,
}
SCALE_CODES = {
    "0-20人": 301,
    "20-99人": 302,
    "100-499人": 303,
    "500-999人": 304,
    "1000-9999人": 305,
    "10000人以上": 306,
}


def _find_code(text: str, mapping: dict[str, int]) -> int | str:
    for label, code in mapping.items():
        if label in text:
            return code
    return ""


def build_graph(browser: BrowserManager):
    """构建职位搜索状态图。每个节点只负责一个明确的状态转换。"""

    def analyze_intent(state: AgentState) -> dict[str, Any]:
        """从用户输入中提取职位关键词和可识别的筛选条件。"""
        text = state.user_input.strip()
        city = ""
        for candidate in browser_city_names(browser):
            if candidate in text:
                city = candidate
                break

        query = text
        for token in (city, "搜索", "查找", "招聘", "职位", "岗位"):
            query = query.replace(token, " ")
        query = re.sub(r"(3K以下|3-5K|5-10K|10-20K|20-50K|50K以上)", " ", query)
        query = re.sub(r"(在校生|应届生|经验不限|1年以内|1-3年|3-5年|5-10年|10年以上)", " ", query)
        query = re.sub(r"(初中及以下|中专/中技|高中|大专|本科|硕士|博士)", " ", query)
        query = re.sub(r"(0-20人|20-99人|100-499人|500-999人|1000-9999人|10000人以上)", " ", query)
        query = re.sub(r"\s+", " ", query).strip() or text
        params = {
            "query": query,
            "city": city,
            "salary_code": _find_code(text, SALARY_CODES),
            "experience_code": _find_code(text, EXPERIENCE_CODES),
            "degree_code": _find_code(text, DEGREE_CODES),
            "scale_code": _find_code(text, SCALE_CODES),
        }
        return {
            "intent": "job_search",
            "search_params": params,
            "status": "intent_analyzed",
        }

    def analyze_resume(state: AgentState) -> dict[str, Any]:
        """简历分析占位节点，后续接入简历解析模型。"""
        return {"resume_analysis": "", "status": "resume_analysis_pending"}

    def init_browser(state: AgentState) -> dict[str, Any]:
        """启动浏览器并打开 Boss 首页。"""
        browser.goto("https://www.zhipin.com/")
        return {"browser": True, "status": "browser_ready"}

    def search_jobs(state: AgentState) -> dict[str, Any]:
        """调用驱动层按意图筛选职位。"""
        params = state.search_params
        jobs = browser.get_job_list(
            params.get("query", ""),
            city=params.get("city", ""),
            salary_code=params.get("salary_code", ""),
            experience_code=params.get("experience_code", ""),
            scale_code=params.get("scale_code", ""),
            degree_code=params.get("degree_code", ""),
        )
        return {
            "jobs": jobs,
            "result": f"找到 {len(jobs)} 个职位",
            "status": "completed",
            "working": False,
        }

    def fail_task(state: AgentState, error: Exception) -> dict[str, Any]:
        return {"error": str(error), "status": "failed", "working": False}

    def run_search(state: AgentState) -> dict[str, Any]:
        try:
            return search_jobs(state)
        except Exception as exc:
            return fail_task(state, exc)

    graph = StateGraph(AgentState)
    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("analyze_resume", analyze_resume)
    graph.add_node("init_browser", init_browser)
    graph.add_node("search_jobs", run_search)
    graph.add_edge(START, "analyze_intent")
    graph.add_edge("analyze_intent", "analyze_resume")
    graph.add_edge("analyze_resume", "init_browser")
    graph.add_edge("init_browser", "search_jobs")
    graph.add_edge("search_jobs", END)
    return graph.compile()


def browser_city_names(browser: BrowserManager) -> list[str]:
    """读取驱动层城市字典，避免在状态图中维护第二份城市表。"""
    from browser import CITY_CODES

    return sorted(CITY_CODES, key=len, reverse=True)


def run_task(user_input: str, browser: BrowserManager, resume: str = "") -> AgentState:
    """执行一次任务并返回最终状态。"""
    state = AgentState(
        task_id=str(uuid4()),
        user_input=user_input,
        jian_li=resume,
        working=True,
        status="started",
    )
    result = build_graph(browser).invoke(state)
    return AgentState.model_validate(result)
