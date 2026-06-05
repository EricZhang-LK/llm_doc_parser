# tests/test_qdrant_store.py
"""QdrantVectorStore 契约测试（内存模式，无需外部服务）。"""
from __future__ import annotations

import pytest

from llm_doc_parser.models import ChunkType, DocumentChunk
from llm_doc_parser.vectorstore.qdrant_store import QdrantVectorStore


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(collection_name="test_docs", dimensions=3)


def _make_chunk(content: str, **metadata: str | int | float) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        chunk_type=ChunkType.TEXT,
        token_count=len(content),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_upsert_and_search(store: QdrantVectorStore) -> None:
    chunks = [_make_chunk("Python is great"), _make_chunk("Rust is fast")]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    await store.upsert(chunks, vectors)
    results = await store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.content == "Python is great"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_empty_store_returns_empty(store: QdrantVectorStore) -> None:
    results = await store.search([1.0, 0.0, 0.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_upsert_length_mismatch_raises(store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        await store.upsert([_make_chunk("a")], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.mark.asyncio
async def test_delete_all_clears_store(store: QdrantVectorStore) -> None:
    await store.upsert([_make_chunk("test")], [[1.0, 0.0, 0.0]])
    await store.delete_all()
    results = await store.search([1.0, 0.0, 0.0])
    assert results == []


@pytest.mark.asyncio
async def test_metadata_filter(store: QdrantVectorStore) -> None:
    chunks = [
        _make_chunk("doc a", source="a"),
        _make_chunk("doc b", source="b"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]

    await store.upsert(chunks, vectors)
    results = await store.search(
        [1.0, 0.0, 0.0],
        top_k=5,
        filters={"source": "a"},
    )

    assert len(results) == 1
    assert results[0].chunk.content == "doc a"
