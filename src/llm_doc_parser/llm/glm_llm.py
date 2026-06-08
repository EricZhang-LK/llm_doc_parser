# src/llm_doc_parser/llm/glm_llm.py
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from llm_doc_parser.llm.base import BaseLLM

# 智谱 OpenAI 兼容接口：https://open.bigmodel.cn/dev/api#openai-sdk
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ZhipuGLM(BaseLLM):
    """
    智谱 GLM 适配器。
    通过 OpenAI 兼容 API 调用 glm-4-flash / glm-4-plus 等模型。
    """

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        base_url: str = ZHIPU_BASE_URL,
    ) -> None:
        self._model = model
        self._temperature = temperature
        resolved_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not resolved_key:
            msg = (
                "ZHIPU_API_KEY is required. "
                "Pass api_key or set the ZHIPU_API_KEY environment variable."
            )
            raise ValueError(msg)
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)

    async def chat(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
