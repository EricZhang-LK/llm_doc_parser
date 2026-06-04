# src/llm_doc_parser/vectorstore/base.py
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from llm_doc_parser.models import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """检索结果，与具体向量库解耦"""
    chunk: DocumentChunk
    score: float          # 相似度分数，越高越相关（归一化到 [0,1]）
    metadata: dict[str, Any]  # 向量库附加元数据（如 distance, id）


class BaseVectorStore(abc.ABC):
    """
    向量存储抽象基类。
    所有向量数据库（Milvus, Qdrant, InMemory）必须实现此接口。
    """

    @abc.abstractmethod
    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """
        批量写入/更新向量及关联文档。
        
        契约保证：
        - chunks[i] 对应 vectors[i]
        - 幂等操作：相同 content 重复写入不会产生重复记录
        """
        ...

    @abc.abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        向量相似度检索。
        
        Returns:
            按 score 降序排列的结果列表
        """
        ...

    @abc.abstractmethod
    async def delete_all(self) -> None:
        """清空存储，用于测试和重置"""
        ...