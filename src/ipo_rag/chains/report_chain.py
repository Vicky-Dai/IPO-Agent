"""Structured IPO report generation with section-targeted retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from ipo_rag.chains.prompts import REPORT_SYSTEM_PROMPT
from ipo_rag.index.retriever import search_documents


@dataclass(frozen=True)
class ReportSectionSpec:
    heading: str
    filter_section: str | None
    queries: tuple[str, ...]


REPORT_SECTIONS: tuple[ReportSectionSpec, ...] = (
    ReportSectionSpec(
        heading="公司与本次发行",
        filter_section="company_offering",
        queries=(
            "公司名称 上市地 板块 保荐人 联席保荐人 所得款项用途 募资用途 时间表",
            "全球发售 香港公开发售 国际发售 上市时间表 保荐人",
        ),
    ),
    ReportSectionSpec(
        heading="业务与行业",
        filter_section="business_industry",
        queries=(
            "主营业务 产品 服务 商业模式 收入来源 客户 行业",
            "公司业务 产品解决方案 客户 收入结构 供应商",
        ),
    ),
    ReportSectionSpec(
        heading="行业地位与前景",
        filter_section="market_position",
        queries=(
            "行业地位 市场规模 增速 排名 市场份额 弗若斯特沙利文",
            "竞争格局 行业排名 市场份额 可比公司 行业前景",
        ),
    ),
    ReportSectionSpec(
        heading="财务与盈利能力",
        filter_section="financials",
        queries=(
            "收入 毛利率 净利润 现金流 财务资料 经调整利润 资产负债表",
            "营收 净亏损 毛利 经营活动现金流 财务表现",
        ),
    ),
    ReportSectionSpec(
        heading="估值与发行条款",
        filter_section="valuation_terms",
        queries=(
            "发售价 市值 市盈率 市销率 估值 全球发售 发售股份",
            "每股发售价 发行市值 估值倍数 公开发售 国际发售",
        ),
    ),
    ReportSectionSpec(
        heading="风险因素",
        filter_section="risk_factors",
        queries=(
            "风险因素 主要风险 监管风险 客户集中 经营风险 财务风险",
            "主要风险 不确定性 竞争风险 合规风险",
        ),
    ),
    ReportSectionSpec(
        heading="打新相关信息",
        filter_section="ipo_trading",
        queries=(
            "基石投资者 超额配售权 绿鞋 回拨 公开发售 国际发售",
            "基石 配售 回拨机制 超额配售权 香港公开发售 国际发售",
        ),
    ),
)


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Document] = []
    for doc in documents:
        key = (
            doc.metadata.get("doc_id"),
            doc.metadata.get("page"),
            doc.page_content[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _format_document(doc: Document) -> str:
    page = doc.metadata.get("page", "?")
    section = doc.metadata.get("section", "general")
    title_hint = doc.metadata.get("title_hint")
    header = f"[page={page} section={section}"
    if title_hint:
        header += f" title={title_hint}"
    header += "]"
    return header + "\n" + doc.page_content


def _search_with_optional_filter(retriever, query: str, *, filter_section: str | None) -> list[Document]:
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is None:
        return retriever.invoke(query)

    docs: list[Document] = []
    if filter_section:
        docs.extend(search_documents(vectorstore, query, filter={"section": filter_section}))
    if len(_dedupe_documents(docs)) < 2:
        docs.extend(search_documents(vectorstore, query))
    return _dedupe_documents(docs)


def _build_section_contexts(retriever) -> str:
    blocks: list[str] = []
    for spec in REPORT_SECTIONS:
        docs: list[Document] = []
        for query in spec.queries:
            docs.extend(
                _search_with_optional_filter(
                    retriever,
                    query,
                    filter_section=spec.filter_section,
                )
            )
        section_docs = _dedupe_documents(docs)[:6]
        if not section_docs:
            body = "未检索到可靠片段。"
        else:
            body = "\n\n".join(_format_document(doc) for doc in section_docs)
        blocks.append(f"### {spec.heading}\n{body}")
    return "\n\n".join(blocks)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _run_report(inputs: dict[str, Any], *, retriever, llm: ChatOpenAI) -> dict[str, str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REPORT_SYSTEM_PROMPT
                + "\n\n以下是按章节检索到的招股书片段。优先使用对应章节的片段；若证据不足，明确写缺失，不要拿其他章节的模糊表述强行补齐。\n\n{section_contexts}",
            ),
            ("human", "{input}"),
        ]
    )
    section_contexts = _build_section_contexts(retriever)
    messages = prompt.invoke(
        {
            "input": inputs["input"],
            "section_contexts": section_contexts,
        }
    ).to_messages()
    answer = llm.invoke(messages)
    return {"answer": _message_text(answer)}


def build_report_chain(retriever, llm: ChatOpenAI) -> Runnable:
    return RunnableLambda(lambda inputs: _run_report(inputs, retriever=retriever, llm=llm))
