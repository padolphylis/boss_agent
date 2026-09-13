from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchParams(BaseModel):
    """搜索参数。"""
    zhi_wei:  Optional[str] = Field(None, description="职位名称")
    money:  Optional[str] = Field(None, description="薪资范围，如：1000-2000")
    experience:  Optional[str] = Field(None, description="经验要求，如：1-3年")
    scale:  Optional[str] = Field(None, description="公司规模，如：10人以下")
    degree:  Optional[str] = Field(None, description="学历要求，如：本科、硕士")
    job_type:  Optional[str] = Field(None, description="岗位类型，如：全职、兼职")
    location:  list[str] = Field(default_factory= list, description="工作地点，如：北京、上海")
    exclude_location:  list[str] = Field(default_factory= list, description="排除的工作地点，如：北京、上海")
    

class AgentState(BaseModel):
    """LangGraph 在各节点之间传递的统一状态。"""

    working: bool = False                                                   #是否正在处理任务      
    task_id: str = ""                                                       #任务ID                     
    error: str = ""                                                         #错误信息                       
    result: str = ""                                                        #llm结果
    status: str = ""                                                        #当前状态
    user_input: str = ""                                                    #用户输入                                                                
    search_params: SearchParams = Field(default_factory=SearchParams)       #搜索参数(意图)
    resume_analysis: str = ""                                               #简历分析内容
    resume_path: str = ""                                                   #简历文件路径
    jobs: list[dict] = Field(default_factory=list)                          #搜索到的职位列表
    job_cards: list[dict] = Field(default_factory=list)                     #职位详情卡片
    matched_jobs: list[dict] = Field(default_factory=list)                  #向量匹配结果
    push_results: list[dict] = Field(default_factory=list)                  #投递结果
    intent: str = ""                                                        #意图标识(job_search/keyword_search)
    browser: bool = False                                                   #浏览器是否可用
    llm_status: bool = False                                                #llm是否可用


  
    