"""Load a prospectus PDF into LangChain Documents (per-page metadata)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf_documents(path: Path) -> list[Document]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "source": str(path),
                    "filename": path.name,
                },
            )
        )
    if not docs:
        raise ValueError(f"No extractable text in PDF: {path}")
    return docs
