from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """LangGraph 在各节点之间传递的统一状态。"""

    working: bool = False
    task_id: str = ""
    error: str = ""
    result: str = ""
    user_input: str = ""
    intent: str = ""
    search_params: dict[str, Any] = Field(default_factory=dict)
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    jian_li: str = ""
    resume_analysis: str = ""
    zhi_wei: str = ""
    zhi_wei_nei_rong: str = ""
    online: bool = False
    browser: bool = False
    status: str = "idle"
