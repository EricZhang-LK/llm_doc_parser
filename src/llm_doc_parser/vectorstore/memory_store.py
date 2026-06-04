# src/llm_doc_parser/vectorstore/memory_store.py
from __future__ import annotations

from typing import Any

import numpy as np

from llm_doc_parser.models import DocumentChunk
from llm_doc_parser.vectorstore.base import BaseVectorStore, SearchResult


class InMemoryVectorStore(BaseVectorStore):
    """
    基于 NumPy 的内存向量存储。
    仅用于单元测试和本地开发验证，不用于生产。
    
    使用余弦相似度，暴力搜索 O(N)。
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: np.ndarray | None = None

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
                )

        new_vectors = np.array(vectors, dtype=np.float32)

        # 简单去重：基于 content hash（生产环境应用唯一 ID）
        existing_contents = {c.content for c in self._chunks}
        for i, chunk in enumerate(chunks):
            if chunk.content not in existing_contents:
                self._chunks.append(chunk)
                if self._vectors is None:
                    self._vectors = new_vectors[i:i+1]
                else:
                    self._vectors = np.vstack([self._vectors, new_vectors[i:i+1]])

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)

        # 余弦相似度 = dot(a,b) / (||a|| * ||b||)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(q)
        similarities = np.dot(self._vectors, q) / np.where(norms == 0, 1, norms)

        # 取 top_k 并按分数降序
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(similarities[idx]),
                metadata={"index": int(idx)},
            ))
        return results

    async def delete_all(self) -> None:
        self._chunks.clear()
        self._vectors = None