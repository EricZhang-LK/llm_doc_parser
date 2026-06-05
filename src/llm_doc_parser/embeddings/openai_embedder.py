# src/llm_doc_parser/embeddings/openai_embedder.py
from __future__ import annotations

from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_doc_parser.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embedding 生产级适配器。
    内置重试、批处理分片、维度控制。
    """

    # OpenAI embeddings API 单次最大输入条数
    _MAX_BATCH_SIZE = 2048

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)  # 我们自己管理重试
        self._max_retries = max_retries

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # 🔥 自动分片：将大批次切分为合规子批次
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[i : i + self._MAX_BATCH_SIZE]
            embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        # ✅ 契约保证：返回顺序与输入严格一致
        assert len(all_embeddings) == len(texts)
        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """带重试的单批次 API 调用"""
        response = await self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._dimensions,
        )
        # OpenAI SDK 保证 data 列表顺序与 input 一致
        return [item.embedding for item in response.data]