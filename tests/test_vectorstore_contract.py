# tests/test_vectorstore_contract.py
"""
向量存储契约测试。
任何 BaseVectorStore 的实现都必须通过这些测试。
未来添加 Qdrant/Milvus 适配器时，复用此测试套件。
"""
from __future__ import annotations

import pytest

from llm_doc_parser.models import ChunkType, DocumentChunk
from llm_doc_parser.vectorstore.memory_store import InMemoryVectorStore


@pytest.fixture
def store():
    return InMemoryVectorStore()


def _make_chunk(content: str) -> DocumentChunk:
    return DocumentChunk(
        content=content, 
        chunk_type=ChunkType.TEXT,
        token_count=len(content)
        )


@pytest.mark.asyncio
async def test_upsert_and_search(store):
    chunks = [_make_chunk("Python is great"), _make_chunk("Rust is fast")]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    await store.upsert(chunks, vectors)
    results = await store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.content == "Python is great"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_empty_store_returns_empty(store):
    results = await store.search([1.0, 0.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_upsert_length_mismatch_raises(store):
    with pytest.raises(ValueError, match="Length mismatch"):
        await store.upsert([_make_chunk("a")], [[1.0], [2.0]])


@pytest.mark.asyncio
async def test_delete_all_clears_store(store):
    await store.upsert([_make_chunk("test")], [[1.0, 0.0]])
    await store.delete_all()
    results = await store.search([1.0, 0.0])
    assert results == []