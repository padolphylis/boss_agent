from pydantic import BaseModel, Field


class AgentState(BaseModel):
    working: bool = False
    task_id: str = ""
    error: str = ""
    result: str = ""
    jian_li: str = ""
    qiu_zhi_mu_biao: str = ""
    zhi_wei: list[str] = Field(default_factory=list)
    zhi_wei_nei_rong: list[str] = Field(default_factory=list)
    zhi_wei_pai_chu: list[str] = Field(default_factory=list)
    



class BrowserState(BaseModel):
    working: bool = False
    task_id: str = ""
    error: str = ""
    result: str = ""
