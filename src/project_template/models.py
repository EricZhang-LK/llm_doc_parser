"""Pydantic v2 model examples for template users."""

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    """Minimal metadata contract example."""

    source_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
