from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from agent import BatterySchedulingAgent
from config.config import llm  # 你的 LLM 配置
import os

app = FastAPI()
agent = BatterySchedulingAgent(llm)

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join("templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    answer = agent.process_query(query)
    return JSONResponse({"answer": answer})