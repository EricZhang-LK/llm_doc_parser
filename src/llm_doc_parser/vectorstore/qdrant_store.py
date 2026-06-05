# src/llm_doc_parser/vectorstore/qdrant_store.py
from __future__ import annotations

import hashlib
from typing import Any, Protocol, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models.models import Condition
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from llm_doc_parser.models import ChunkType, DocumentChunk
from llm_doc_parser.vectorstore.base import BaseVectorStore, SearchResult


class _QdrantClient(Protocol):
    """qdrant-client 异步接口子集（规避生成代码的类型不完整问题）。"""

    async def get_collections(self) -> Any: ...

    async def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: VectorParams,
    ) -> bool | None: ...

    async def upsert(
        self,
        *,
        collection_name: str,
        points: list[PointStruct],
    ) -> Any: ...

    async def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: Filter | None,
    ) -> list[ScoredPoint]: ...

    async def delete_collection(self, collection_name: str) -> bool | None: ...


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant 生产级向量存储适配器。
    支持本地内存模式和远程服务器模式。
    """

    def __init__(
        self,
        collection_name: str,
        dimensions: int,
        url: str | None = None,       # None = 本地内存模式
        api_key: str | None = None,
    ) -> None:
        self._collection = collection_name
        self._dimensions = dimensions
        # url=None 时使用内存模式，便于测试；生产环境传入 Qdrant 服务地址
        if url is None:
            client: _QdrantClient = AsyncQdrantClient(location=":memory:")
        else:
            client = AsyncQdrantClient(url=url, api_key=api_key)
        self._client = client
        self._initialized = False

    async def _ensure_collection(self) -> None:
        """懒初始化集合，避免构造函数中执行异步操作"""
        if self._initialized:
            return
        collections = [
            c.name for c in (await self._client.get_collections()).collections
            ]
        if self._collection not in collections:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions, distance=Distance.COSINE
                    ),
            )
        self._initialized = True

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Length mismatch: {len(chunks)} vs {len(vectors)}")
        await self._ensure_collection()

        points = [
            PointStruct(
                # 🔥 基于 content 生成确定性 ID，保证幂等性
                id=self._content_id(chunk.content),
                vector=vector,
                payload={
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "token_count": chunk.token_count,
                    **{f"meta_{k}": v for k, v in chunk.metadata.items()},
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        await self._ensure_collection()

        qdrant_filter = self._build_filter(filters) if filters else None
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = DocumentChunk(
                content=payload["content"],
                chunk_type=ChunkType(payload["chunk_type"]),
                token_count=payload["token_count"],
                metadata={
                    k.replace("meta_", "", 1): v
                    for k, v in payload.items()
                    if k.startswith("meta_")
                },
            )
            results.append(SearchResult(
                chunk=chunk,
                score=hit.score,
                metadata={"id": str(hit.id)},
            ))
        return results

    async def delete_all(self) -> None:
        await self._ensure_collection()
        await self._client.delete_collection(self._collection)
        self._initialized = False

    @staticmethod
    def _content_id(content: str) -> str:
        """基于内容生成确定性 UUID（32 位 hex），保证幂等且兼容 Qdrant 本地模式"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """将通用 dict filter 翻译为 Qdrant Filter DSL"""
        conditions = [
            FieldCondition(
                key=f"meta_{key}",
                match=MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return Filter(must=cast(list[Condition], conditions))