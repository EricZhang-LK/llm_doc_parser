# src/llm_doc_parser/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLM(ABC):
    """
    LLM 抽象基类。
    定义了同步问答和异步流式问答的标准接口。
    """

    @abstractmethod
    async def chat(self, prompt: str) -> str:
        """
        普通问答模式：接收 Prompt，返回完整答案。
        """
        pass

    @abstractmethod
    def chat_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        流式问答模式：接收 Prompt，异步 yield 每一个生成的文本片段（token）。
        """
        raise NotImplementedError