# src/llm_doc_parser/rag_pipeline.py
from __future__ import annotations

from collections.abc import AsyncGenerator

from llm_doc_parser.embeddings.base import BaseEmbedder
from llm_doc_parser.llm.base import BaseLLM
from llm_doc_parser.prompt_engine import PromptEngine
from llm_doc_parser.vectorstore.base import BaseVectorStore


class RAGPipeline:
    """
    RAG 核心编排器。
    负责将 Embedding、VectorStore 和 LLM 串联起来，完成检索增强生成。
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm_client: BaseLLM,
        prompt_engine: PromptEngine | None = None,
        top_k: int = 3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._prompt_engine = prompt_engine or PromptEngine()
        self._top_k = top_k

    async def query(self, question: str) -> str:
        """
        执行一次完整的 RAG 问答流程。
        
        流程：
        1. Query Embedding
        2. Vector Search
        3. Prompt Rendering
        4. LLM Generation
        """
        # 1. 将问题转化为向量
        query_vector = (await self._embedder.embed_texts([question]))[0]

        # 2. 检索相关文档
        search_results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=self._top_k,
        )

        if not search_results:
            return "抱歉，未在知识库中找到相关信息。"

        # 3. 渲染 Prompt
        # 将 SearchResult 转换为 Prompt 需要的格式
        formatted_context = [
            {"content": res.chunk.content, 
            "metadata": res.chunk.metadata, 
            "score": res.score}
            for res in search_results
        ]
        
        prompt = self._prompt_engine.render(
            "rag_qa.jinja2",
            question=question,
            context=formatted_context,
        )

        # 4. 调用 LLM 生成答案
        # 这里假设 llm_client 有一个 chat 方法，接收 prompt 返回文本
        # 具体的 LLM 抽象将在 D7 完善
        return await self._llm_client.chat(prompt)
    
    async def query_stream(self, question: str) -> AsyncGenerator[str, None]:
        """
        流式 RAG 问答流程。
        """
        # 1. 向量检索 (与非流式相同)
        query_vector = (await self._embedder.embed_texts([question]))[0]
        search_results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=self._top_k,
        )

        if not search_results:
            yield "抱歉，未在知识库中找到相关信息。"
            return

        # 2. 渲染 Prompt (与非流式相同)
        formatted_context = [
            {"content": res.chunk.content,
            "metadata": res.chunk.metadata,
            "score": res.score}
            for res in search_results
        ]
        prompt = self._prompt_engine.render(
            "rag_qa.jinja2",
            question=question,
            context=formatted_context,
        )

        # 3. 🔥 调用 LLM 的流式接口，并逐字 yield 出去
        async for token in self._llm_client.chat_stream(prompt):
            yield token