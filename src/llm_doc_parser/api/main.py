# src/llm_doc_parser/api/main.py
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llm_doc_parser.api.schemas import ChatRequest, ChatResponse
from llm_doc_parser.embeddings.base import BaseEmbedder
from llm_doc_parser.embeddings.openai_embedder import OpenAIEmbedder
from llm_doc_parser.embeddings.zhipu_embedder import ZhipuEmbedder
from llm_doc_parser.llm.base import BaseLLM
from llm_doc_parser.llm.glm_llm import ZhipuGLM
from llm_doc_parser.llm.openai_llm import OpenAILLM
from llm_doc_parser.rag_pipeline import RAGPipeline
from llm_doc_parser.vectorstore.qdrant_store import QdrantVectorStore

# 1. 初始化 FastAPI 应用
app = FastAPI(title="LLM Doc Parser RAG API", version="1.0.0")

# 配置跨域（CORS），允许前端本地调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 懒加载初始化 RAG 核心组件（避免启动时阻塞）
rag_pipeline: RAGPipeline | None = None

def _create_embedder() -> BaseEmbedder:

    return ZhipuEmbedder(
        model=os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3"),
        dimensions=int(os.getenv("ZHIPU_EMBEDDING_DIMENSIONS", "1024")),
        api_key=os.getenv("ZHIPU_API_KEY"),
    )
    # return OpenAIEmbedder(
    #     model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    # )


def _create_llm() -> BaseLLM:

    return ZhipuGLM(
        model=os.getenv("GLM_MODEL", "glm-4-flash"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        )
    # return OpenAILLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        embedder = _create_embedder()
        vector_store = QdrantVectorStore(
            collection_name="documents",
            dimensions=embedder.dimensions,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        llm = _create_llm()
        rag_pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            llm_client=llm,
            top_k=3
        )
    return rag_pipeline

# 3. 非流式问答接口
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    pipeline = get_rag_pipeline()
    try:
        # 这里为了演示简单，直接返回答案。实际生产中可解析 sources
        answer = await pipeline.query(request.question)
        return ChatResponse(answer=answer, sources=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 流式问答接口（核心亮点）
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    pipeline = get_rag_pipeline()
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in pipeline.query_stream(request.question):
                # SSE 协议格式：data: 内容\n\n
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)