# src/llm_doc_parser/api/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """用户提问请求体"""
    question: str = Field(..., description="用户的问题", min_length=1)
    top_k: int = Field(3, description="检索相关文档的数量", ge=1, le=10)

class ChatResponse(BaseModel):
    """非流式回答响应体"""
    answer: str = Field(..., description="LLM 生成的最终答案")
    sources: list[str] = Field(default_factory=list, description="参考文档的来源列表")