# src/llm_doc_parser/parsers/docling_parser.py
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from llm_doc_parser.models import ChunkType, DocumentChunk, ParseResult
from llm_doc_parser.parsers.base import BaseParser


class DoclingParser(BaseParser):
    """
    Docling 解析器适配器。
    将 Docling 的原生输出转换为我们的 ParseResult 契约。
    """

    def __init__(self, enable_ocr: bool = True) -> None:
        # 预配置解析选项，避免每次 parse() 都重建
        pipeline_options = PdfPipelineOptions(do_ocr=enable_ocr)
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    async def parse(self, file_path: Path) -> ParseResult:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        start = time.perf_counter()

        # 🔥 关键：将 CPU 密集的同步调用卸载到线程池
        # 如果不这样做，FastAPI 的事件循环会被阻塞数秒甚至数十秒
        result = await asyncio.to_thread(self._convert_sync, file_path)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 将 Docling 原生结果转换为我们的强类型契约
        chunks = self._extract_chunks(result)
        warnings: list[str] = []
        if not chunks:
            warnings.append(f"No content extracted from {file_path.name}")

        return ParseResult(
            source_file=str(file_path),
            chunks=chunks,
            parse_time_ms=elapsed_ms,
            warnings=warnings,
        )

    def _convert_sync(self, file_path: Path) -> ConversionResult:
        """同步转换方法，供 to_thread 调用"""
        return self._converter.convert(str(file_path))

    def _extract_chunks(self, docling_result: ConversionResult) -> list[DocumentChunk]:
        """
        从 Docling 结果中提取分块，映射到 DocumentChunk 契约。
        这里是适配器模式的核心：翻译两种数据模型。
        """
        chunks: list[DocumentChunk] = []
        doc = docling_result.document

        for element in doc.texts:
            # 跳过空文本
            text = element.text.strip()
            if not text:
                continue

            # 根据 Docling 的 label 映射到我们的 ChunkType
            chunk_type = self._map_chunk_type(element.label)

            page_no = element.prov[0].page_no if element.prov else 0
            chunks.append(DocumentChunk(
                content=text,
                chunk_type=chunk_type,
                metadata={
                    "page": page_no,
                    "label": str(element.label),
                },
                token_count=len(text) // 4,  # 粗略估算，W3 会替换为精确 tokenizer
            ))

        return chunks

    @staticmethod
    def _map_chunk_type(label: object) -> ChunkType:
        """Docling label → ChunkType 映射表"""
        mapping = {
            "title": ChunkType.TEXT,
            "section_header": ChunkType.TEXT,
            "paragraph": ChunkType.TEXT,
            "text": ChunkType.TEXT,
            "table": ChunkType.TABLE,
            "formula": ChunkType.FORMULA,
            "caption": ChunkType.IMAGE_CAPTION,
        }
        label_key = getattr(label, "value", label)
        return mapping.get(str(label_key).lower(), ChunkType.TEXT)