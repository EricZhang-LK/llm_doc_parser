# src/llm_doc_parser/tokenizer.py
from __future__ import annotations

import tiktoken


class TokenCounter:
    """
    精确 Token 计数器。
    封装 tiktoken，提供统一的计数接口，隔离具体编码格式的变化。
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        # cl100k_base 覆盖 GPT-4/GPT-3.5/Claude 等主流模型
        # 缓存编码器实例，避免重复加载（tiktoken 初始化有开销）
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """精确计算文本的 token 数"""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        在 token 级别安全截断文本。
        注意：这是 token 级截断，不保证句子完整性。
        仅用于兜底场景，正常分块应使用 SemanticChunker。
        """
        if self.count(text) <= max_tokens:
            return text
        tokens = self._encoding.encode(text)[:max_tokens]
        return self._encoding.decode(tokens)