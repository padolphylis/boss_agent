import json

from .llm import create_llm
from .models import (
    ActionType,
    AgentAction,
    Job,
    JobAnalysis,
    JobTarget,
    PageObservation,
)


SYSTEM_PROMPT = """你是职位分析助手。只能分析职位并提出受限动作，不能编造简历信息。
最终必须返回 JSON，字段为：recommended、score、matched_items、missing_items、risks、reason、message。
不要返回 submit_application 动作；推荐投递时，下一步必须等待人工确认。"""


def build_analysis_prompt(job: Job, target: JobTarget, resume_text: str) -> str:
    return json.dumps(
        {
            "job": job.model_dump(),
            "target": target.model_dump(),
            "resume": resume_text,
            "instruction": "根据职位、求职目标和简历判断是否推荐。score 必须是 0 到 100 的整数。",
        },
        ensure_ascii=False,
    )


def analyze_job(job: Job, target: JobTarget, resume_text: str) -> JobAnalysis:
    llm = create_llm().with_structured_output(JobAnalysis)
    return llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", build_analysis_prompt(job, target, resume_text)),
    ])


def observe_page(observation: PageObservation) -> str:
    """合并 DOM 和 OCR 文本，供职位解析流程使用。"""
    parts = [observation.dom_text.strip(), observation.ocr_text.strip()]
    return "\n".join(part for part in parts if part)


def action_for_analysis(job: Job, analysis: JobAnalysis) -> AgentAction:
    if not analysis.recommended:
        return AgentAction(
            action=ActionType.SKIP_JOB,
            job_id=job.job_id,
            reason=analysis.reason,
        )
    return AgentAction(
        action=ActionType.WAIT_FOR_USER,
        job_id=job.job_id,
        reason=analysis.reason,
        message=analysis.message,
    )
