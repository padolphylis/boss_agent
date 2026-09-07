from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from body import run_task
from browser import BrowserManager


browser = BrowserManager()
browser_lock = Lock()


class SearchRequest(BaseModel):
    """前端提交的任务参数。"""

    message: str = Field(min_length=1, max_length=500)
    resume: str = Field(default="", max_length=20000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用退出时关闭浏览器，避免留下 Chromium 进程。"""
    yield
    browser.close()


app = FastAPI(title="职位助手", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """返回单页前端。"""
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """执行一次 LangGraph 任务并返回结构化状态。"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="请输入职位搜索需求。")

    with browser_lock:
        state = run_task(request.message, browser, request.resume)

    # 简历分析节点目前是占位实现，原始简历已随状态保存。
    return state.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
