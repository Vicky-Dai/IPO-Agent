"""OpenAI embeddings wrapper."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from ipo_rag.config import openai_api_key


def get_embeddings(*, model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, api_key=openai_api_key())
