from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    zhi_wei: Optional[str] = Field(None, description="用户希望搜索的职位名称，不要包含排除项")
    money: Optional[str] = Field(
        None,
        description="薪资码表名称，只能使用 money_codes.json 中的选项，如：5-10K、10-20K",
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
    job_type: Optional[str] = Field(None, description="岗位类型，如：全职、兼职")
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
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="用户明确不想做的行业、岗位或工作内容关键词，如：漫画、销售",
    )

    @field_validator("experience", "scale", "degree", "stage", mode="before")
    @classmethod
    def normalize_multi_select(cls, value):
        """兼容模型返回单字符串，并统一为多选列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class AgentState(BaseModel):
    """LangGraph 在各节点之间传递的统一状态。"""

    working: bool = False                                                   #是否正在处理任务      
    task_id: str = ""                                                       #任务ID                     
    error: str = ""                                                         #错误信息                       
    result: str = ""                                                        #llm结果
    status: str = ""                                                        #当前状态
    user_input: str = ""                                                    #用户输入
    original_input: str = ""                                                #首次用户输入
    pending_question: str = ""                                              #待补充问题
    search_params: SearchParams = Field(default_factory=SearchParams)       #搜索参数(意图)
    resume_analysis: str = ""                                               #简历分析内容
    resume_path: str = ""                                                   #简历文件路径
    resume_filename: str = ""                                               #用户上传的简历文件名
    resume_parse_status: str = "none"                                       #简历解析状态
    jobs: list[dict] = Field(default_factory=list)                          #搜索到的职位列表
    job_cards: list[dict] = Field(default_factory=list)                     #职位详情卡片
    matched_jobs: list[dict] = Field(default_factory=list)                  #向量匹配结果
    push_results: list[dict] = Field(default_factory=list)                  #投递结果
    pipeline_processed: bool = False                                       #职位管线是否已完成抓取、匹配及后续处理
    checkpoint_node: str = ""                                            #最近完成的流程节点
    intent: str = ""                                                        #意图标识
    browser: bool = False                                                   #浏览器是否可用
    llm_status: bool = False                                                #llm是否可用
    match_query: str = ""                                                   #用于向量匹配的补充查询，推荐时可包含简历内容
    pipeline_warning: str = ""                                              #职位管线降级或超时提示
    conversation_history: list[dict[str, str]] = Field(default_factory=list) #当前会话的历史消息
