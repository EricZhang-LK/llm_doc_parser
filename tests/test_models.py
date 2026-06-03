# tests/test_models.py
import pytest
from pydantic import ValidationError

from llm_doc_parser.models import DocumentChunk


def test_valid_chunk():
    chunk = DocumentChunk(content="Hello", token_count=1)
    assert chunk.content == "Hello"

def test_empty_content_fails():
    with pytest.raises(ValidationError, match="whitespace"):
        DocumentChunk(content="   ", token_count=0)