from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from analysis_resume import ResumeAnalyzer
from browser import BrowserManager
from job import search_jobs as run_search_jobs
from state import AgentState, SearchParams
from task_control import TaskControl


class ReadResumeInput(BaseModel):
    """读取简历 Tool 的输入。"""

    model_config = ConfigDict(extra="forbid")

    resume_path: str = Field(
        min_length=1,
        description="本机已保存的 PDF、DOCX 或 DOC 简历路径",
    )


class CollectJobCandidatesInput(BaseModel):
    """职位候选采集 Tool 的输入。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        default="",
        description="当前后台任务 ID，用于日志和任务控制",
    )
    search_params: SearchParams = Field(
        description="已经校验过的职位搜索条件",
    )
    retrieval_query: str = Field(
        default="",
        description="由后端根据结构化条件和简历画像生成的正向语义查询",
    )
    user_input: str = Field(
        default="",
        description="当前任务的原始用户输入，仅作为旧逻辑的最终查询回退",
    )


def create_read_resume_tool(
    analyzer_factory: Callable[..., ResumeAnalyzer] = ResumeAnalyzer,
    *,
    max_chars: int | None = None,
    timeout: float | None = None,
) -> StructuredTool:
    """创建简历读取 Tool，允许服务端注入解析器实现和解析限制。"""

    def read_resume(resume_path: str) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if max_chars is not None:
            options["max_chars"] = max_chars
        if timeout is not None:
            options["timeout"] = timeout
        result = analyzer_factory(resume_path, **options).analyze()
        return result.to_dict()

    return StructuredTool.from_function(
        func=read_resume,
        name="read_resume",
        description=(
            "读取本机已保存的 PDF、DOCX 或 DOC 简历并提取文本。"
            "只用于解析简历，不负责生成匹配结论。"
        ),
        args_schema=ReadResumeInput,
    )


read_resume_tool = create_read_resume_tool()


def create_collect_job_candidates_tool(
    browser: BrowserManager,
    browser_io_lock,
    task_control: TaskControl | None = None,
    runner: Callable[..., dict[str, Any]] = run_search_jobs,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> StructuredTool:
    """创建绑定服务端运行时依赖的职位候选采集 Tool。"""

    def collect_job_candidates(
        task_id: str = "",
        search_params: SearchParams | dict[str, Any] | None = None,
        retrieval_query: str = "",
        user_input: str = "",
    ) -> dict[str, Any]:
        params = (
            search_params
            if isinstance(search_params, SearchParams)
            else SearchParams.model_validate(search_params or {})
        )
        state = AgentState(
            task_id=task_id,
            intent=params.intent,
            user_input=user_input,
            search_params=params,
            retrieval_query=retrieval_query,
            match_query=retrieval_query,
        )
        options = {}
        if progress_callback is not None:
            options["on_progress"] = progress_callback
        return runner(browser, state, browser_io_lock, task_control, **options)

    return StructuredTool.from_function(
        func=collect_job_candidates,
        name="collect_job_candidates",
        description=(
            "根据已经校验的职位搜索条件，使用本机已登录的招聘网站浏览器"
            "读取职位列表和详情，并返回候选职位及匹配前的召回结果。"
            "不要把该工具当作最终投递工具；它不会执行投递。"
        ),
        args_schema=CollectJobCandidatesInput,
    )
