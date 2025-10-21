from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from agent import BatterySchedulingAgent
from config.config import llm  # 你的 LLM 配置
import os
import json
import asyncio

app = FastAPI()
agent = BatterySchedulingAgent(llm)

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join("templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/ask1")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    answer = agent.process_query(query)
    return JSONResponse({"answer": answer})

@app.post("/ask")
async def ask(request: Request):
    """流式输出接口"""
    data = await request.json()
    query = data.get("query", "")

    async def generate():
        try:
            print(f"开始流式处理查询: {query}")
            async for chunk in agent.process_query_stream(query):
                # print(f"发送chunk: {chunk[:50]}...")  # 打印前50个字符
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            
            print("流式处理完成")
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"流式处理错误: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用nginx缓冲
            }
        )
