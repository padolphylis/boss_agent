import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchParams(BaseModel):
    """从自然语言中提取的职位搜索参数。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal[
        "job_search",
        "job_recommendation",
        "resume_analysis",
        "job_analysis",
        "chat",
        "unclear",
    ] = Field(
        "job_search",
        description="用户意图：职位搜索、职位推荐、简历分析、职位分析、闲聊，或信息不足需要澄清",
    )
    reply: Optional[str] = Field(
        None,
        description="闲聊回复或澄清问题；职位搜索时为空",
    )
    resume_matching: Literal["use", "ignore"] = Field(
        "use",
        description=(
            "后端解析后的最终简历匹配策略：use 或 ignore。"
            "该字段由后端结合本轮 resume_matching_intent 和会话记忆确定，"
            "模型不要根据默认值自行覆盖会话偏好。"
        ),
    )
    resume_matching_intent: Literal["use", "ignore", "unspecified"] = Field(
        "unspecified",
        description=(
            "本轮用户对简历匹配策略的明确意图。use=明确要求结合简历，"
            "ignore=明确要求不参考简历，unspecified=本轮没有改变策略。"
            "只根据当前用户消息判断，不要把会话历史中的旧偏好重复当作本轮指令。"
        ),
    )
    zhi_wei: list[str] = Field(
        default_factory=list,
        description=(
            "用户希望搜索的职位名称列表，不要包含排除项；"
            "多个职位会分别参与城市、求职类型和薪资的组合搜索"
        ),
    )
    salary: list[str] = Field(
        default_factory=list,
        description=(
            "用户真实的薪资要求，只保存用户实际表达的范围，不受 BOSS 码表限制；"
            "例如用户说 7-8K 时返回 ['7-8K']，用户说 7K 以上时返回 ['7K以上']。"
            "该字段用于前端展示和职位详情硬性过滤，不直接传给 BOSS 搜索接口。"
        ),
    )
    money: list[str] = Field(
        default_factory=list,
        description=(
            "BOSS 平台筛选薪资桶，可多选；每个选项只能使用 money_codes.json 中的完整名称，"
            "如：5-10K、10-20K。它只是扩大搜索召回范围，不能替代 salary；"
            "有 salary 时由后端根据真实薪资范围推导，执行搜索时每次只传一个薪资区间"
        ),
    )
    experience: list[str] = Field(
        default_factory=list,
        description="经验码表名称，可多选，只能使用 experience_codes.json 中的选项",
    )
    scale: list[str] = Field(
        default_factory=list,
        description="公司规模码表名称，可多选，只能使用 scale_codes.json 中的选项",
    )
    degree: list[str] = Field(
        default_factory=list,
        description="学历码表名称，可多选，只能使用 degree_codes.json 中的选项",
    )
    job_type: list[str] = Field(
        default_factory=list,
        description=(
            "求职类型，可多选，只能使用岗位类型码表中的完整名称，如：全职、兼职；"
            "执行搜索时每次只传一个求职类型"
        ),
    )
    stage: list[str] = Field(default_factory=list, description="阶段筛选项，可多选")
    location: list[str] = Field(default_factory=list, description="工作地点，如：北京、上海")
    exclude_location: list[str] = Field(
        default_factory=list,
        description="明确排除的工作地点，如：北京、上海",
    )
    location_unlimited: bool = Field(
        False,
        description="用户明确表示地点不限（如：全国都可以、去哪都行）时为 True",
    )
    boss_activity: Literal["any", "online", "recent"] = Field(
        "any",
        description=(
            "BOSS 活跃度意图：any=未限定，online=只要当前在线，"
            "recent=要求最近一段时间内活跃。'最近活跃'但未给出天数时仍设为 recent，"
            "由后续流程追问，不要自行猜测天数"
        ),
    )
    boss_activity_days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description=(
            "当 boss_activity=recent 时表示最近多少天内活跃，例如 3；"
            "用户只说'最近活跃'但没有具体天数时必须为 null"
        ),
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="用户明确不想做的行业、岗位或工作内容关键词，如：漫画、销售",
    )
    retrieval_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "用于语义匹配的岗位技术、领域和能力关键词，如：Python、FastAPI、RAG、Agent；"
            "不要放地点、薪资、经验、学历、公司规模或排除项"
        ),
    )
    retrieval_responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "用于语义匹配的核心工作内容，如：后端服务开发、接口开发、模型接入、知识库建设；"
            "只保留正向职责，不要放排除项"
        ),
    )

    @field_validator(
        "experience",
        "scale",
        "degree",
        "stage",
        "money",
        "job_type",
        "retrieval_keywords",
        "retrieval_responsibilities",
        "salary",
        mode="before",
    )
    @classmethod
    def normalize_multi_select(cls, value):
        """兼容模型返回单字符串，并统一为多选列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("zhi_wei", mode="before")
    @classmethod
    def normalize_positions(cls, value):
        """兼容旧快照中的单字符串，并清洗职位列表中的空项和重复项。"""
        if value is None:
            return []
        values = [value] if isinstance(value, str) else value
        result = []
        seen = set()
        for item in values or []:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_activity_field(cls, value):
        """兼容早期快照中的 boss_online_only，不把旧字段暴露给新意图 schema。"""
        if not isinstance(value, dict) or "boss_online_only" not in value:
            return value
        migrated = dict(value)
        legacy_online = migrated.pop("boss_online_only")
        migrated.setdefault(
            "boss_activity",
            "online" if legacy_online else "any",
        )
        return migrated


def _clean_retrieval_items(values: list[str] | str | None) -> list[str]:
    """清洗检索字段，避免空值、重复项和异常空白污染查询向量。"""
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def build_retrieval_query(
    params: SearchParams,
    *,
    resume_profile: str = "",
) -> str:
    """从结构化需求生成正向语义查询。"""
    lines = []
    roles = _clean_retrieval_items(params.zhi_wei)
    if roles:
        lines.append(f"目标岗位：{'、'.join(roles)}")

    excluded = {
        item.casefold()
        for item in _clean_retrieval_items(params.exclude_keywords)
    }
    keywords = [
        item for item in _clean_retrieval_items(params.retrieval_keywords)
        if not any(excluded_item in item.casefold() for excluded_item in excluded)
    ]
    if keywords:
        lines.append(f"核心技术与能力：{'、'.join(keywords)}")

    responsibilities = [
        item
        for item in _clean_retrieval_items(params.retrieval_responsibilities)
        if not any(excluded_item in item.casefold() for excluded_item in excluded)
    ]
    if responsibilities:
        lines.append(f"核心工作内容：{'、'.join(responsibilities)}")

    profile = re.sub(r"\s+", " ", str(resume_profile or "")).strip()
    if profile:
        lines.append(f"候选人匹配画像：{profile[:4000]}")

    return "\n".join(lines)[:16000]


class AgentState(BaseModel):
    """LangGraph 在各节点之间传递的统一状态。"""

    working: bool = False
    task_id: str = ""
    error: str = ""
    error_source: str = ""
    result: str = ""
    status: str = ""
    user_input: str = ""
    original_input: str = ""
    pending_question: str = ""
    search_params: SearchParams = Field(default_factory=SearchParams)
    resume_analysis: str = ""
    resume_path: str = ""
    resume_filename: str = ""
    resume_parse_status: str = "none"
    jobs: list[dict] = Field(default_factory=list)
    job_cards: list[dict] = Field(default_factory=list)
    matched_jobs: list[dict] = Field(default_factory=list)
    semantic_results: list[dict] = Field(default_factory=list)
    semantic_evaluated: bool = False
    push_results: list[dict] = Field(default_factory=list)
    delivery_processed: bool = False
    pipeline_processed: bool = False
    checkpoint_node: str = ""
    intent: str = ""
    browser: bool = False
    llm_status: bool = False
    retrieval_query: str = ""
    match_query: str = ""
    resume_match_profile: str = ""
    pipeline_warning: str = ""
    pipeline_progress: dict[str, Any] = Field(default_factory=dict)
    conversation_preferences: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
