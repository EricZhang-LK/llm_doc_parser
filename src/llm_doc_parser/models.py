# src/llm_doc_parser/models.py
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE_CAPTION = "image_caption"

class DocumentChunk(BaseModel):
    content: str = Field(..., min_length=1)
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)
    token_count: int = Field(ge=0)

    @field_validator("content")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Chunk content cannot be empty or whitespace-only")
        return cleaned

class ParseResult(BaseModel):
    source_file: str
    chunks: list[DocumentChunk]
    parse_time_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)