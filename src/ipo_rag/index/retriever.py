"""Retriever factory."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from ipo_rag.config import report_fetch_k, report_retrieval_k, top_k


def get_retriever(
    vectorstore: Chroma,
    k: int | None = None,
) -> VectorStoreRetriever:
    k = k if k is not None else top_k()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search_documents(
    vectorstore: Chroma,
    query: str,
    *,
    k: int | None = None,
    filter: dict[str, str] | None = None,
) -> list[Document]:
    k = k if k is not None else report_retrieval_k()
    fetch_k = max(report_fetch_k(), k * 3)
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        filter=filter,
    )
    if docs:
        return docs
    return vectorstore.similarity_search(query, k=k, filter=filter)
