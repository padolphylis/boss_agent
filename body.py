from langgraph.graph import StateGraph, State, END
from langchain_openai import OpenAI
from state import AgentState
import os
import load_dotenv
load_dotenv()


llm = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("openai_base_url"),
    model=os.getenv("model"),
)

def chu_shi_hua(state: AgentState):
    """初始化状态"""
    if
    return state

def check_state(state: AgentState):
    """检查状态"""
    return state


def check_state():
    """检查状态"""
    return State


