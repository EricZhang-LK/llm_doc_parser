# tests/test_docling_parser.py
from __future__ import annotations

import pytest
from pathlib import Path

from llm_doc_parser.parsers.docling_parser import DoclingParser
from llm_doc_parser.models import ChunkType


# 准备一个真实的测试 PDF（放在 tests/fixtures/ 下）
FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURE_DIR / "sample.pdf"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not found")
@pytest.mark.asyncio
async def test_docling_parser_returns_valid_result():
    # sample.pdf 为扫描件，必须开启 OCR 才能提取文本
    parser = DoclingParser(enable_ocr=True)
    result = await parser.parse(SAMPLE_PDF)

    # 验证契约完整性
    assert result.source_file == str(SAMPLE_PDF)
    assert result.parse_time_ms > 0
    assert len(result.chunks) > 0

    # 验证每个 chunk 都符合 DocumentChunk 契约
    for chunk in result.chunks:
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0
        assert isinstance(chunk.chunk_type, ChunkType)
        assert chunk.token_count >= 0


@pytest.mark.asyncio
async def test_docling_parser_file_not_found():
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError):
        await parser.parse(Path("/nonexistent/file.pdf"))