"""Load a prospectus PDF into LangChain Documents (per-page metadata)."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


SECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("risk_factors", ("风险因素", "主要风险", "风险")),
    (
        "ipo_trading",
        ("基石投资者", "基石", "超额配售权", "超额配股权", "绿鞋", "回拨", "重新分配"),
    ),
    (
        "valuation_terms",
        ("发售价", "发售股份", "全球发售", "香港公开发售", "国际发售", "市盈率", "市销率", "估值"),
    ),
    (
        "financials",
        ("财务资料", "综合损益", "资产负债表", "现金流量", "收入", "毛利", "净利润", "经调整"),
    ),
    (
        "market_position",
        ("市场份额", "市场排名", "排名", "份额", "弗若斯特沙利文", "竞争格局", "竞争"),
    ),
    (
        "business_industry",
        ("业务", "产品", "服务", "解决方案", "客户", "供应商", "行业", "商业模式"),
    ),
    (
        "company_offering",
        ("概览", "公司资料", "股份代号", "保荐人", "董事", "上市", "所得款项用途", "募集资金用途"),
    ),
)


def _make_doc_id(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or path.stem.lower()


def _title_hint(text: str) -> str:
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if 2 <= len(line) <= 60:
            return line
    return ""


def _infer_section(text: str) -> str:
    sample = text[:4000]
    best_section = "general"
    best_score = 0
    for section, keywords in SECTION_RULES:
        score = sum(sample.count(keyword) for keyword in keywords)
        if score > best_score:
            best_section = section
            best_score = score
    return best_section


def load_pdf_documents(path: Path) -> list[Document]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = PdfReader(str(path))
    doc_id = _make_doc_id(path)
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
                    "doc_id": doc_id,
                    "page": i + 1,
                    "section": _infer_section(text),
                    "source": str(path),
                    "filename": path.name,
                    "title_hint": _title_hint(text),
                },
            )
        )
    if not docs:
        raise ValueError(f"No extractable text in PDF: {path}")
    return docs
