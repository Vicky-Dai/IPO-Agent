"""Multi-turn conversational RAG (history-aware retrieval + message history)."""

from __future__ import annotations

from collections.abc import Callable

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from ipo_rag.chains.prompts import CHAT_SYSTEM_WITH_CONTEXT, CONTEXTUALIZE_Q_SYSTEM


def build_history_aware_retriever(llm: ChatOpenAI, retriever):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def build_chat_rag_chain(llm: ChatOpenAI, retriever):
    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHAT_SYSTEM_WITH_CONTEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def wrap_with_message_history(
    rag_chain: Runnable,
    get_session_history: Callable[[str], BaseChatMessageHistory],
) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
