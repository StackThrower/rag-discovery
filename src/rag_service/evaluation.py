"""Offline evaluation. Without measurement, every RAG "improvement" is a guess.

* Retrieval: hit rate@k and MRR — does the right document reach the generator, and how high?
  Cheap (no LLM), run it on every change of chunking / embeddings / reranker.
* Answers: the expected document is cited, required facts are present, no false refusals.
  For LLM-as-judge metrics (faithfulness, answer relevancy) see RAGAS: https://docs.ragas.io/
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from rag_service.retrieval import Retriever
from rag_service.service import RagService


class EvalItem(BaseModel):
    question: str
    expected_files: list[str]
    must_contain: list[str] = Field(default_factory=list)


class RetrievalRow(BaseModel):
    question: str
    rank: int | None
    retrieved: list[str]


class RetrievalReport(BaseModel):
    k: int
    hit_rate: float
    mrr: float
    rows: list[RetrievalRow]


def load_eval(path: Path) -> list[EvalItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalItem.model_validate(json.loads(line)) for line in lines if line.strip()]


def evaluate_retrieval(retriever: Retriever, items: list[EvalItem]) -> RetrievalReport:
    rows = []
    for item in items:
        files = [c.file_name for c in retriever.search(item.question)]
        rank = next((i for i, f in enumerate(files, 1) if f in item.expected_files), None)
        rows.append(RetrievalRow(question=item.question, rank=rank, retrieved=files))
    return RetrievalReport(
        k=retriever.top_k,
        hit_rate=mean(r.rank is not None for r in rows) if rows else 0.0,
        mrr=mean(1 / r.rank if r.rank else 0.0 for r in rows) if rows else 0.0,
        rows=rows,
    )


class AnswerRow(BaseModel):
    question: str
    cited_expected: bool
    facts_present: bool
    refused: bool


class AnswerReport(BaseModel):
    citation_accuracy: float
    """Share of answers citing at least one expected document."""
    fact_recall: float
    """Share of answers containing all `must_contain` strings."""
    false_refusal_rate: float
    """Share of answerable questions the system refused."""
    rows: list[AnswerRow]


def evaluate_answers(service: RagService, items: list[EvalItem]) -> AnswerReport:
    """End-to-end check (calls the LLM). Each question runs in a fresh thread."""
    rows = []
    for item in items:
        ans = service.ask(item.question)
        cited = {c.location.split(",")[0] for c in ans.citations}
        text = ans.answer.casefold()
        rows.append(
            AnswerRow(
                question=item.question,
                cited_expected=bool(cited & set(item.expected_files)),
                facts_present=all(f.casefold() in text for f in item.must_contain),
                refused=ans.insufficient_context,
            )
        )
    n = len(rows) or 1
    return AnswerReport(
        citation_accuracy=sum(r.cited_expected for r in rows) / n,
        fact_recall=sum(r.facts_present for r in rows) / n,
        false_refusal_rate=sum(r.refused for r in rows) / n,
        rows=rows,
    )
