"""Single-shot IPO report RAG chain."""

from __future__ import annotations

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from ipo_rag.chains.prompts import REPORT_SYSTEM_PROMPT


def build_report_chain(retriever, llm: ChatOpenAI) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REPORT_SYSTEM_PROMPT + "\n\n以下为检索到的招股书片段（含页码），请据此撰写报告：\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    combine_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs)
