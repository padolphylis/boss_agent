from pydantic import BaseModel

class AgentState(BaseModel):
    working: bool = False
    task_id: str = ""
    error: str = ""
    result: str = ""


class BrowserState(BaseModel):
    working: bool = False
    task_id: str = ""
    error: str = ""
    result: str = ""
