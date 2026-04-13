"""Typer CLI: ingest, report, chat, eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ipo_rag.chains.chat_chain import build_chat_rag_chain, wrap_with_message_history
from ipo_rag.chains.report_chain import build_report_chain
from ipo_rag.config import (
    chroma_persist_dir,
    embedding_model_name,
    get_chat_llm,
    top_k,
)
from ipo_rag.index.embeddings import get_embeddings
from ipo_rag.index.retriever import get_retriever
from ipo_rag.index.vector_store import load_vectorstore
from ipo_rag.ingestion.pipeline import ingest_pdf
from ipo_rag.memory.session_store import get_session_history

app = typer.Typer(help="IPO prospectus RAG (minimal: OpenAI + Chroma + LangChain)")


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, help="Prospectus PDF path"),
    collection: str = typer.Option(
        "default", "--collection", "-c", help="Chroma collection name"
    ),
) -> None:
    """Parse PDF, chunk, embed into local Chroma."""
    n, pdir = ingest_pdf(pdf, collection)
    typer.echo(f"Ingested {n} chunks → {pdir} (collection={collection!r})")


def _load_rag_stack(collection: str):
    emb = get_embeddings(model=embedding_model_name())
    vs = load_vectorstore(
        persist_directory=chroma_persist_dir(),
        collection_name=collection,
        embeddings=emb,
    )
    retriever = get_retriever(vs)
    llm = get_chat_llm()
    return retriever, llm


@app.command("report")
def report_cmd(
    collection: str = typer.Option("default", "--collection", "-c"),
) -> None:
    """Generate a Markdown report from retrieved prospectus chunks."""
    retriever, llm = _load_rag_stack(collection)
    chain = build_report_chain(retriever, llm)
    out = chain.invoke(
        {
            "input": "请根据检索到的招股书片段生成一份中文 Markdown 分析报告，章节与要求见系统提示。"
        }
    )
    typer.echo(out["answer"])


@app.command()
def chat(
    collection: str = typer.Option("default", "--collection", "-c"),
    session: str = typer.Option("cli", "--session", "-s", help="Session id for memory"),
) -> None:
    """Interactive multi-turn Q&A over the prospectus (type 'exit' to quit)."""
    retriever, llm = _load_rag_stack(collection)
    base = build_chat_rag_chain(llm, retriever)
    chain = wrap_with_message_history(base, get_session_history)
    typer.echo(
        "多轮对话已启动（session=%s）。输入 exit / quit 结束。\n" % session
    )
    while True:
        try:
            q = input("你: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        result = chain.invoke(
            {"input": q},
            config={"configurable": {"session_id": session}},
        )
        typer.echo("\n助手:\n" + result["answer"] + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@app.command("eval")
def eval_benchmark(
    gold: Optional[Path] = typer.Option(
        None,
        "--gold",
        "-g",
        help="Path to gold_qa.jsonl (default: <repo>/benchmarks/gold_qa.jsonl)",
        exists=False,
        dir_okay=False,
    ),
    collection: str = typer.Option("default", "--collection", "-c"),
) -> None:
    """Compute recall@k: whether any gold page appears in top-k retrieved chunks."""
    path = gold or (_repo_root() / "benchmarks" / "gold_qa.jsonl")
    if not path.is_file():
        typer.echo(f"Gold file not found: {path}", err=True)
        raise typer.Exit(code=1)
    retriever, _ = _load_rag_stack(collection)
    k = top_k()
    hits = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = row["question"]
        gold_pages = row.get("gold_pages") or []
        docs = retriever.invoke(q)
        pages = {d.metadata.get("page") for d in docs if d.metadata.get("page") is not None}
        ok = bool(set(gold_pages) & pages)
        hits += int(ok)
        total += 1
        typer.echo(f"Q: {q[:60]}... | hit={ok} | gold_pages={gold_pages} | retrieved_pages={sorted(pages)[:10]}")
    if total:
        typer.echo(f"\nRecall@{k} (page hit rate): {hits}/{total} = {hits/total:.2%}")
    else:
        typer.echo("No rows in gold file.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
