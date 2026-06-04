# src/llm_doc_parser/embeddings/base.py
from __future__ import annotations

import abc


class BaseEmbedder(abc.ABC):
    """
    Embedding 模型抽象基类。
    所有 Embedding 提供商（OpenAI, BGE, Cohere）必须实现此接口。
    """

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """返回向量维度，用于初始化向量存储"""
        ...

    @abc.abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化。
        
        契约保证：
        - 返回顺序与输入严格一致
        - 每个向量长度 == self.dimensions
        - 空列表输入 → 空列表输出
        
        Raises:
            EmbeddingError: API 调用失败（非重试性错误）
        """
        ...