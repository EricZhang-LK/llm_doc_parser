# tests/test_llm_glm.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_doc_parser.llm.glm_llm import ZhipuGLM


@pytest.mark.asyncio
async def test_zhipu_glm_chat() -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "你好，我是 GLM。"

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    llm = ZhipuGLM(api_key="test-key")
    llm._client = mock_client

    answer = await llm.chat("你好")

    assert answer == "你好，我是 GLM。"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "glm-4-flash"
    assert call_kwargs["messages"] == [{"role": "user", "content": "你好"}]


@pytest.mark.asyncio
async def test_zhipu_glm_chat_stream() -> None:
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices[0].delta.content = "流式"
    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices[0].delta.content = "输出"

    async def mock_stream():
        yield mock_chunk_1
        yield mock_chunk_2

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream()

    llm = ZhipuGLM(api_key="test-key", model="glm-4-plus")
    llm._client = mock_client

    full_answer = ""
    async for token in llm.chat_stream("测试"):
        full_answer += token

    assert full_answer == "流式输出"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["model"] == "glm-4-plus"


def test_zhipu_glm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
        ZhipuGLM(api_key=None)
