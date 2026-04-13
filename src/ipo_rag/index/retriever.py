"""Retriever factory."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from ipo_rag.config import top_k


def get_retriever(
    vectorstore: Chroma,
    k: int | None = None,
) -> VectorStoreRetriever:
    k = k if k is not None else top_k()
    return vectorstore.as_retriever(search_kwargs={"k": k})
