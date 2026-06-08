# src/llm_doc_parser/embeddings/zhipu_embedder.py
from __future__ import annotations

import os

from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_doc_parser.embeddings.base import BaseEmbedder

# 智谱 OpenAI 兼容接口：https://open.bigmodel.cn/dev/api#openai-sdk
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ZhipuEmbedder(BaseEmbedder):
    """
    智谱 Embedding 适配器。
    支持 embedding-3（可调维度 256–2048）与 embedding-2。
    """

    # 智谱单次请求最多 64 条文本
    _MAX_BATCH_SIZE = 64

    def __init__(
        self,
        model: str = "embedding-3",
        dimensions: int = 1024,
        api_key: str | None = None,
        base_url: str = ZHIPU_BASE_URL,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        resolved_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not resolved_key:
            msg = (
                "ZHIPU_API_KEY is required. "
                "Pass api_key or set the ZHIPU_API_KEY environment variable."
            )
            raise ValueError(msg)
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[i : i + self._MAX_BATCH_SIZE]
            embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        assert len(all_embeddings) == len(texts)
        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
