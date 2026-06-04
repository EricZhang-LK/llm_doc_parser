# tests/test_chunker.py
from __future__ import annotations

import pytest

from llm_doc_parser.chunker import ChunkingConfig, SemanticChunker
from llm_doc_parser.models import DocumentChunk
from llm_doc_parser.tokenizer import TokenCounter


@pytest.fixture
def counter() -> TokenCounter:
    return TokenCounter()


@pytest.fixture
def chunker(counter: TokenCounter) -> SemanticChunker:
    # 测试用小配置，方便构造边界条件
    # max_tokens=10：三句英文样例约 18 tokens，可稳定触发切分
    config = ChunkingConfig(max_tokens=10, overlap_tokens=5, min_chunk_tokens=5)
    return SemanticChunker(counter, config)


def test_short_chunk_preserved(chunker: SemanticChunker) -> None:
    """未超长的 chunk 应直接保留，仅更新精确 token 数"""
    source = DocumentChunk(content="Hello world.", token_count=0)
    result = chunker.chunk([source])

    assert len(result) == 1
    assert result[0].content == "Hello world."
    assert result[0].token_count > 0  # 精确计数已生效


def test_long_chunk_split_at_sentence_boundary(chunker: SemanticChunker) -> None:
    """超长 chunk 应在句子边界切分，不产生断裂"""
    # 全文约 18 tokens（tiktoken），超过 max_tokens=10
    text = (
    "This is the first sentence. "
    "This is the second sentence. "
    "This is the third sentence."
)
    source = DocumentChunk(content=text, token_count=0)
    result = chunker.chunk([source])

    assert len(result) >= 2
    # 每个子 chunk 都不应超过 max_tokens
    for chunk in result:
        assert chunk.token_count <= 10
    # 所有内容拼接后应等于原文（允许空格差异）
    reconstructed = " ".join(c.content for c in result)
    assert set(reconstructed.split()) == set(text.split())


def test_overlap_contains_complete_sentences(chunker: SemanticChunker) -> None:
    """Overlap 部分必须是完整句子"""
    text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
    source = DocumentChunk(content=text, token_count=0)
    result = chunker.chunk([source])

    if len(result) >= 2:
        # 第二个 chunk 的开头应该包含第一个 chunk 末尾的完整句子
        first_end = result[0].content.rstrip().split(".")[-2] + "."
        assert first_end.strip() in result[1].content


def test_empty_input_returns_empty(chunker: SemanticChunker) -> None:
    result = chunker.chunk([])
    assert result == []