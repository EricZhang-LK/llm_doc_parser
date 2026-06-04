# src/llm_doc_parser/chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass

from llm_doc_parser.models import ChunkType, DocumentChunk
from llm_doc_parser.tokenizer import TokenCounter


@dataclass(frozen=True)
class ChunkingConfig:
    """分块配置，使用 frozen dataclass 防止运行时意外修改"""
    max_tokens: int = 512
    overlap_tokens: int = 64      # 约 12% 重叠率
    min_chunk_tokens: int = 50    # 过短的 chunk 缺乏语义价值，合并到前一个


# 句子分割正则：支持中英文句号、问号、感叹号及换行符
_SENTENCE_SPLITTER = re.compile(r'(?<=[.!?。！？\n])\s*')


class SemanticChunker:
    """
    语义感知分块器。
    核心原则：结构优先 → 句子边界 → 长度兜底。
    """

    def __init__(
        self,
        counter: TokenCounter,
        config: ChunkingConfig | None = None,
    ) -> None:
        self._counter = counter
        self._config = config or ChunkingConfig()

    def chunk(self, source_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        对 D2 产出的原始分块进行语义级二次分块。
        
        处理流程：
        1. 遍历每个原始 chunk
        2. 若未超长，直接保留（尊重原始结构）
        3. 若超长，按句子边界切分，带 overlap 重组
        4. 过滤掉过短的尾部碎片
        """
        result: list[DocumentChunk] = []

        for source in source_chunks:
            token_count = self._counter.count(source.content)

            # ✅ 未超长：直接保留，保持原始结构完整性
            if token_count <= self._config.max_tokens:
                # 更新精确 token 计数（替换 D2 的估算值）
                result.append(source.model_copy(update={
                    "token_count": token_count,
                }))
                continue

            # ⚠️ 超长：按句子边界切分
            sub_chunks = self._split_with_overlap(
                content=source.content,
                chunk_type=source.chunk_type,
                base_metadata=source.metadata,
            )
            result.extend(sub_chunks)

        return result

    def _split_with_overlap(
        self,
        content: str,
        chunk_type: ChunkType,
        base_metadata: dict[str, str | int | float],
    ) -> list[DocumentChunk]:
        """按句子边界切分，带 overlap 重组"""
        sentences = _SENTENCE_SPLITTER.split(content)
        sentences = [s for s in sentences if s.strip()]  # 过滤空白句

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._counter.count(sentence)

            # 单句就超长：强制截断（极端情况兜底）
            if sentence_tokens > self._config.max_tokens:
                # 先把当前累积的句子输出为一个 chunk
                if current_sentences:
                    chunks.append(self._build_chunk(
                        current_sentences, chunk_type, base_metadata,
                    ))
                    current_sentences = []
                    current_tokens = 0

                # 对超长句子做 token 级截断
                truncated = self._counter.truncate(sentence, self._config.max_tokens)
                chunks.append(DocumentChunk(
                    content=truncated,
                    chunk_type=chunk_type,
                    metadata={**base_metadata, "truncated": True},
                    token_count=self._counter.count(truncated),
                ))
                continue

            # 加入当前句后是否超长？
            if current_tokens + sentence_tokens > self._config.max_tokens:
                # 输出当前累积的 chunk
                chunks.append(self._build_chunk(
                    current_sentences, chunk_type, base_metadata,
                ))

                # 🔥 Overlap 逻辑：从已输出的 chunk 末尾回溯句子
                # 直到 overlap 达到目标值或句子用完
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for prev_sentence in reversed(current_sentences):
                    prev_tokens = self._counter.count(prev_sentence)
                    if overlap_tokens + prev_tokens > self._config.overlap_tokens:
                        break
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_tokens += prev_tokens

                # 新 chunk 以 overlap 句子开头
                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # 处理剩余句子
        if current_sentences:
            last_chunk = self._build_chunk(current_sentences, chunk_type, base_metadata)
            # 过滤过短的尾部碎片：合并到前一个 chunk
            if (
                last_chunk.token_count < self._config.min_chunk_tokens
                and chunks
            ):
                merged_content = chunks[-1].content + " " + last_chunk.content
                chunks[-1] = chunks[-1].model_copy(update={
                    "content": merged_content,
                    "token_count": self._counter.count(merged_content),
                })
            else:
                chunks.append(last_chunk)

        return chunks

    def _build_chunk(
        self,
        sentences: list[str],
        chunk_type: ChunkType,
        base_metadata: dict[str, str | int | float],
    ) -> DocumentChunk:
        """从句子列表构建 DocumentChunk"""
        content = "".join(sentences).strip()
        return DocumentChunk(
            content=content,
            chunk_type=chunk_type,
            metadata=base_metadata.copy(),  # 防止多个 chunk 共享同一 metadata 引用
            token_count=self._counter.count(content),
        )