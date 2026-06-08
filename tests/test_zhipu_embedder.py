# tests/test_zhipu_embedder.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_doc_parser.embeddings.zhipu_embedder import ZhipuEmbedder


@pytest.mark.asyncio
async def test_zhipu_embedder_returns_vectors() -> None:
    mock_item_1 = MagicMock()
    mock_item_1.embedding = [0.1, 0.2, 0.3]
    mock_item_2 = MagicMock()
    mock_item_2.embedding = [0.4, 0.5, 0.6]
    mock_response = MagicMock()
    mock_response.data = [mock_item_1, mock_item_2]

    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_response

    embedder = ZhipuEmbedder(api_key="test-key", dimensions=3)
    embedder._client = mock_client

    vectors = await embedder.embed_texts(["你好", "世界"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["model"] == "embedding-3"
    assert call_kwargs["dimensions"] == 3
    assert call_kwargs["input"] == ["你好", "世界"]


@pytest.mark.asyncio
async def test_zhipu_embedder_empty_input() -> None:
    embedder = ZhipuEmbedder(api_key="test-key")
    assert await embedder.embed_texts([]) == []


def test_zhipu_embedder_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
        ZhipuEmbedder(api_key=None)
