from io import BytesIO
from pathlib import Path

from docx import Document

import src.document_loader as document_loader


def test_extract_text_from_txt() -> None:
    text = document_loader.extract_text_from_upload("sample.txt", b"Hello world")
    assert text == "Hello world"


def test_extract_text_from_docx_bytes() -> None:
    doc = Document()
    doc.add_paragraph("First line")
    doc.add_paragraph("Second line")
    buffer = BytesIO()
    doc.save(buffer)

    text = document_loader.extract_text_from_upload("sample.docx", buffer.getvalue())
    assert text == "First line\nSecond line"


def test_extract_text_from_pdf_uses_pypdf_reader(monkeypatch) -> None:
    class FakePage:
        def __init__(self, value: str) -> None:
            self._value = value

        def extract_text(self) -> str:
            return self._value

    class FakeReader:
        def __init__(self, file_obj: BytesIO) -> None:
            assert hasattr(file_obj, "read")
            self.pages = [FakePage("Page one"), FakePage("Page two")]

    monkeypatch.setattr(document_loader, "PdfReader", FakeReader)

    text = document_loader.extract_text_from_upload("sample.pdf", b"%PDF-FAKE")
    assert text == "Page one\nPage two"


def test_extract_text_for_real_rules_docx() -> None:
    project_root = Path(__file__).resolve().parents[1]
    doc_path = project_root / "ClassifyingRules.docx"
    text = document_loader.extract_text_from_upload(doc_path.name, doc_path.read_bytes())

    assert "RESTRICTED" in text
    assert "CONFIDENTIAL" in text
    assert "PUBLIC" in text
