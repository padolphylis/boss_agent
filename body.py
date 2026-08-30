from langgraph.graph import StateGraph, State, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAI
from state import AgentState,InitState
from jsonschema import validate, ValidationError
import os
import load_dotenv
load_dotenv()

zhi_wei_schema = {
    "type": "object",
    "properties": {
        "职位": {"type": "string"},
        "薪资": {"type": "integer",
                'min':0,# 最小薪资
                'max':50000,# 最大薪资
        "工作地点": {"type": "string"},
        "岗位内容": {"type": "string"},
        "非岗位内容": {"type": "string"},
    },
    "required": ["职位", "薪资", "工作地点", "岗位内容", "非岗位内容"],
    "additionalProperties": False
    }
}


llm = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("openai_base_url"),
    model=os.getenv("model"),
    response_format="json_object",
)

def chu_shi_hua(state: InitState) -> InitState:
    """初始化状态"""
    system_msg = SystemMessage(content= """你是一个信息提取助手。请分析用户的消息，提取与工作岗位相关的信息，并严格按照以下要求返回 JSON 对象。

要求：
- 只返回一个 JSON 对象，不要包含任何解释、标记、代码块或额外文字。
- JSON 必须包含且仅包含以下字段：
  - "职位"：用户期望的职位名称，类型为字符串。
  - "薪资"：用户期望的薪资（月薪），类型为整数，单位元，数值必须在 0 到 50000 之间（包含边界）。
  - "工作地点"：用户期望的工作地点，类型为字符串。
  - "岗位内容"：用户描述的岗位职责或工作内容，类型为字符串。
  - "非岗位内容"：用户提到的与岗位本身无关的其他要求（如福利、公司文化、通勤时间等），类型为字符串。
- 所有字段都是必需的，不能缺失。
- 不允许添加任何额外字段。

用户消息：{{user_message}}""")
    response = llm.invoke("")
    try:
        ai_msg = AIMessage(content=response.content)
        validate(ai_msg.content, zhi_wei_schema)
    except ValidationError as e:
        state.error = str(e)
        return state
    return state


def check_state(state:  InitState) -> InitState| AgentState:
    """检查状态"""

    return state


def check_state():
    """检查状态"""
    return State


