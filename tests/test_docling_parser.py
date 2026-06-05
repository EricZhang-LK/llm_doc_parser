# tests/test_docling_parser.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_doc_parser.models import ChunkType
from llm_doc_parser.parsers.docling_parser import DoclingParser

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURE_DIR / "sample.pdf"


def _make_docling_result(
    texts: list[tuple[str, str, int]],
) -> SimpleNamespace:
    """构造最小 Docling ConversionResult 替身，避免依赖真实解析。"""
    text_items = [
        SimpleNamespace(
            text=content,
            label=label,
            prov=[SimpleNamespace(page_no=page)],
        )
        for content, label, page in texts
    ]
    return SimpleNamespace(document=SimpleNamespace(texts=text_items))


@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not found")
@pytest.mark.asyncio
async def test_docling_parser_returns_valid_result() -> None:
    """端到端测试：需从 HuggingFace 下载 Docling 模型，仅本地手动运行。"""
    parser = DoclingParser(enable_ocr=True)
    result = await parser.parse(SAMPLE_PDF)

    assert result.source_file == str(SAMPLE_PDF)
    assert result.parse_time_ms > 0
    assert len(result.chunks) > 0

    for chunk in result.chunks:
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0
        assert isinstance(chunk.chunk_type, ChunkType)
        assert chunk.token_count >= 0


@pytest.mark.asyncio
async def test_docling_parser_file_not_found() -> None:
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError):
        await parser.parse(Path("/nonexistent/file.pdf"))


def test_map_chunk_type() -> None:
    assert DoclingParser._map_chunk_type("table") == ChunkType.TABLE
    assert DoclingParser._map_chunk_type("formula") == ChunkType.FORMULA
    assert DoclingParser._map_chunk_type("caption") == ChunkType.IMAGE_CAPTION
    assert DoclingParser._map_chunk_type("unknown_label") == ChunkType.TEXT


def test_extract_chunks_skips_empty_text() -> None:
    parser = DoclingParser()
    result = _make_docling_result([
        ("", "text", 1),
        ("Valid content.", "paragraph", 2),
    ])

    chunks = parser._extract_chunks(result)

    assert len(chunks) == 1
    assert chunks[0].content == "Valid content."
    assert chunks[0].chunk_type == ChunkType.TEXT
    assert chunks[0].metadata == {"page": 2, "label": "paragraph"}


@pytest.mark.asyncio
async def test_parse_uses_mocked_converter(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    parser = DoclingParser()
    mock_result = _make_docling_result([
        ("First paragraph.", "paragraph", 1),
        ("Table cell", "table", 1),
    ])

    with patch.object(parser, "_convert_sync", return_value=mock_result):
        result = await parser.parse(pdf)

    assert result.source_file == str(pdf)
    assert len(result.chunks) == 2
    assert result.chunks[1].chunk_type == ChunkType.TABLE
    assert result.warnings == []
