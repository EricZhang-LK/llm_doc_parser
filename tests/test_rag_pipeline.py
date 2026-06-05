# tests/test_rag_pipeline.py
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from llm_doc_parser.embeddings.base import BaseEmbedder
from llm_doc_parser.models import ChunkType, DocumentChunk
from llm_doc_parser.rag_pipeline import RAGPipeline
from llm_doc_parser.vectorstore.base import BaseVectorStore, SearchResult


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock(spec=BaseEmbedder)
    # 假设问题 "什么是 RAG？" 的向量是 [1.0, 0.0, 0.0]
    embedder.embed_texts.return_value = [[1.0, 0.0, 0.0]]
    return embedder

@pytest.fixture
def mock_vector_store():
    store = AsyncMock(spec=BaseVectorStore)
    # 模拟检索到一条相关文档
    chunk = DocumentChunk(
        content="RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写。",
        chunk_type=ChunkType.TEXT,
        token_count=10,
        metadata={"source": "wiki.txt"}
    )
    store.search.return_value = [SearchResult(chunk=chunk, score=0.95, metadata={})]
    return store

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.return_value = "RAG 是检索增强生成的缩写。"
    return llm

@pytest.mark.asyncio
async def test_rag_pipeline_success(mock_embedder, mock_vector_store, mock_llm):
    pipeline = RAGPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )

    answer = await pipeline.query("什么是 RAG？")

    # 验证流程调用
    mock_embedder.embed_texts.assert_called_once_with(["什么是 RAG？"])
    mock_vector_store.search.assert_called_once()
    mock_llm.chat.assert_called_once()
    
    # 验证最终答案
    assert "检索增强生成" in answer

@pytest.mark.asyncio
async def test_rag_pipeline_no_results(mock_embedder, mock_vector_store, mock_llm):
    # 模拟检索为空
    mock_vector_store.search.return_value = []

    pipeline = RAGPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )

    answer = await pipeline.query("一个不存在的问题")

    assert answer == "抱歉，未在知识库中找到相关信息。"
    # 检索为空时，不应调用 LLM
    mock_llm.chat.assert_not_called()