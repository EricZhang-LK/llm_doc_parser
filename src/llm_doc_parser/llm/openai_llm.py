# src/llm_doc_parser/llm/openai_llm.py
from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from llm_doc_parser.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 适配器。
    支持 GPT-3.5/4/4o 系列模型，并内置流式输出支持。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key)

    async def chat(self, prompt: str) -> str:
        """
        非流式调用：直接返回完整答案。
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        流式调用：逐字（Token）异步产出答案。
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            stream=True,  # 🔥 开启流式模式
        )

        # 遍历异步流，实时提取每个 chunk 的文本内容
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content