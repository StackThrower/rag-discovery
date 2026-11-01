"""CLI: rag ingest | ask | chat | eval | stats | serve"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from rag_service.evaluation import evaluate_answers, evaluate_retrieval, load_eval
from rag_service.schemas import Answer
from rag_service.service import RagService

app = typer.Typer(add_completion=False, help="RAG over your documents (LangGraph + Qdrant).")
console = Console()


def _print_answer(a: Answer) -> None:
    console.print(Markdown(a.answer))
    if a.citations:
        t = Table("#", "source", "snippet", show_lines=False)
        for c in a.citations:
            t.add_row(str(c.n), c.location, c.snippet[:90].replace("\n", " ") + "…")
        console.print(t)
    flags = []
    if a.insufficient_context:
        flags.append("insufficient context")
    if a.rewrites:
        flags.append(f"query rewritten → '{a.search_query}'")
    if flags:
        console.print(f"[dim]({'; '.join(flags)})[/dim]")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Files or folders (.md .txt .pdf)"),
) -> None:
    """Index documents (idempotent: unchanged files are skipped)."""
    r = RagService.from_settings().ingest(paths)
    console.print(f"indexed {len(r.indexed)} files ({r.chunks} chunks), skipped {len(r.skipped)}")
    for src, err in r.failed.items():
        console.print(f"[red]failed {src}: {err}[/red]")


@app.command()
def ask(question: str) -> None:
    """One-off question."""
    _print_answer(RagService.from_settings().ask(question))


@app.command()
def chat() -> None:
    """Multi-turn chat: follow-up questions are rewritten using the history."""
    svc, thread = RagService.from_settings(), str(uuid.uuid4())
    while (q := Prompt.ask("[bold cyan]?[/bold cyan]", default="")).strip():
        _print_answer(svc.ask(q, thread_id=thread))


@app.command("eval")
def evaluate(
    eval_file: Path = typer.Argument(Path("data/eval.jsonl")),
    answers: bool = typer.Option(False, "--answers", help="Also run end-to-end (calls the LLM)"),
) -> None:
    """Retrieval quality (hit rate, MRR) on a labelled set; optionally answer quality too."""
    svc, items = RagService.from_settings(), load_eval(eval_file)
    report = evaluate_retrieval(svc.retriever, items)
    t = Table("question", "rank", "top retrieved")
    for row in report.rows:
        rank = str(row.rank) if row.rank else "[red]miss[/red]"
        t.add_row(row.question[:60], rank, ", ".join(dict.fromkeys(row.retrieved))[:60])
    console.print(t)
    console.print(f"hit_rate@{report.k} = {report.hit_rate:.2f}   MRR = {report.mrr:.2f}")
    if answers:
        a = evaluate_answers(svc, items)
        console.print(
            f"citation accuracy = {a.citation_accuracy:.2f}   fact recall = {a.fact_recall:.2f}"
            f"   false refusals = {a.false_refusal_rate:.2f}"
        )


@app.command()
def stats() -> None:
    """Number of indexed chunks."""
    svc = RagService.from_settings()
    console.print(f"collection '{svc.s.collection}': {svc.store.count()} chunks")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("rag_service.api:app", host=host, port=port)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
