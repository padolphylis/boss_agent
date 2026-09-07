import uvicorn


if __name__ == "__main__":
    # 从 boss_agent 目录执行：py main.py
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
