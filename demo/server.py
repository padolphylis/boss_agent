from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM Backend Demo")

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("openai_base_url"),
    model=os.getenv("model"),
)

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面 - 前端入口"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/llm")
async def llm_endpoint(request: Request):
    """同步调用 LLM（非流式输出）
    适合普通场景，直接返回完整结果。
    """
    data = await request.json()
    prompt = data.get("prompt", "Tell me about AI applications in a short paragraph.")

    completion = client.chat.completions.create(
        model=os.getenv("model"),
        messages=[{"role": "user", "content": prompt}]
    )

    result = completion.choices[0].message.content

    return {"result": result}


@app.post("/llm/stream")
async def llm_stream_endpoint(request: Request):
    """流式调用 LLM（实时输出）
    适合需要流式响应的场景（如聊天界面实时显示）。
    使用 Server-Sent Events (SSE) 实现。
    """
    data = await request.json()
    prompt = data.get("prompt", "Tell me about AI applications in a short paragraph.")

    async def generate_stream():
        try:
            stream = client.chat.completions.create(
                model=os.getenv("model"),
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            yield "data: 正在思考...\n\n"
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)