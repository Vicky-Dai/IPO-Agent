"""Chroma vector store: load or persist."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def persist_documents(
    documents: list[Document],
    *,
    persist_directory: Path,
    collection_name: str,
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def load_vectorstore(
    *,
    persist_directory: Path,
    collection_name: str,
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
