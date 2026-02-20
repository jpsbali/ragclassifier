from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_text_from_upload(filename: str, content_bytes: bytes) -> str:
    lower_name = filename.lower()
    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        return content_bytes.decode("utf-8", errors="ignore").strip()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(content_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if lower_name.endswith(".docx"):
        doc = Document(BytesIO(content_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    # Fallback for unknown file types.
    return content_bytes.decode("utf-8", errors="ignore").strip()

