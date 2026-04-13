"""Ingest PDF → chunks → Chroma."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ipo_rag.config import chunk_overlap, chunk_size, chroma_persist_dir, embedding_model_name
from ipo_rag.index.embeddings import get_embeddings
from ipo_rag.index.vector_store import persist_documents
from ipo_rag.ingestion.chunking import split_documents
from ipo_rag.ingestion.pdf_loader import load_pdf_documents


def ingest_pdf(
    pdf_path: Path,
    collection_name: str,
    *,
    persist_directory: Path | None = None,
) -> tuple[int, Path]:
    """
    Returns (number of chunks, persist directory used).
    """
    persist_directory = persist_directory or chroma_persist_dir()
    raw_docs: list[Document] = load_pdf_documents(Path(pdf_path))
    chunks: list[Document] = split_documents(
        raw_docs, chunk_size=chunk_size(), chunk_overlap=chunk_overlap()
    )
    emb = get_embeddings(model=embedding_model_name())
    persist_documents(
        chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=emb,
    )
    return len(chunks), persist_directory
