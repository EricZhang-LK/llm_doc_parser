# tests/test_llm_openai.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_doc_parser.llm.openai_llm import OpenAILLM


@pytest.mark.asyncio
async def test_openai_llm_chat_stream():
    # 模拟 OpenAI SDK 的流式返回结构
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices[0].delta.content = "RAG"
    
    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices[0].delta.content = " 是检索增强生成。"
    
    # 构造一个异步生成器作为 mock 返回值
    async def mock_stream():
        yield mock_chunk_1
        yield mock_chunk_2

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream()

    # 注入 mock 客户端（这里为了测试方便，直接修改私有属性）
    llm = OpenAILLM(api_key="test-key")
    llm._client = mock_client

    # 收集流式输出的结果
    full_answer = ""
    async for token in llm.chat_stream("什么是RAG？"):
        full_answer += token

    assert full_answer == "RAG 是检索增强生成。"
    # 验证 SDK 的 stream 参数确实被设为了 True
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is True