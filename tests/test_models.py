import pytest
from project_template.models import DocumentMeta
from pydantic import ValidationError


def test_document_meta_accepts_valid_input() -> None:
    doc = DocumentMeta(source_id="doc-1", title="template")
    assert doc.title == "template"


def test_document_meta_requires_title() -> None:
    with pytest.raises(ValidationError):
        DocumentMeta(source_id="doc-1", title="")
