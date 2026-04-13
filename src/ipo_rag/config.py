"""Load settings from environment (.env supported)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root: parent of src/ when running from repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _data_dir() -> Path:
    return Path(os.environ.get("IPO_DATA_DIR", str(_REPO_ROOT / "data")))


def chroma_persist_dir() -> Path:
    p = Path(os.environ.get("IPO_CHROMA_DIR", str(_data_dir() / "chroma")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def chat_model_name() -> str:
    return os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def embedding_model_name() -> str:
    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def chunk_size() -> int:
    return int(os.environ.get("IPO_CHUNK_SIZE", "900"))


def chunk_overlap() -> int:
    return int(os.environ.get("IPO_CHUNK_OVERLAP", "120"))


def top_k() -> int:
    return int(os.environ.get("IPO_TOP_K", "6"))


def get_chat_llm():
    """Chat model for RAG chains (low temperature for factual tasks)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=chat_model_name(),
        api_key=openai_api_key(),
        temperature=0,
    )
