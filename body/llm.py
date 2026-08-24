import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class AgentState(BaseModel):
    history: list[dict] = Field(default_factory=list)


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_URL"),
        model=os.environ["OPENAI_MODEL"],
        temperature=0,
    )

