from __future__ import annotations

from pathlib import Path

from conftest import DATA

from rag_service.evaluation import evaluate_answers, evaluate_retrieval, load_eval
from rag_service.retrieval import Retriever
from rag_service.schemas import GeneratedAnswer


def test_hybrid_retrieval_quality_on_eval_set(make_service, docs_dir: Path) -> None:
    svc, _ = make_service()
    svc.ingest([docs_dir])
    report = evaluate_retrieval(svc.retriever, load_eval(DATA / "eval.jsonl"))
    # even with offline lexical stand-ins for embeddings the right doc is in top-5
    assert report.hit_rate == 1.0
    assert report.mrr >= 0.9


def test_hybrid_beats_dense_only(make_service, docs_dir: Path) -> None:
    items = load_eval(DATA / "eval.jsonl")
    hybrid, _ = make_service()
    hybrid.ingest([docs_dir])
    dense = Retriever(hybrid.store, fetch_k=20, top_k=5)
    dense.store.vs.retrieval_mode = type(hybrid.store.vs.retrieval_mode).DENSE
    mrr_dense = evaluate_retrieval(dense, items).mrr
    dense.store.vs.retrieval_mode = type(hybrid.store.vs.retrieval_mode).HYBRID
    mrr_hybrid = evaluate_retrieval(hybrid.retriever, items).mrr
    assert mrr_hybrid >= mrr_dense


class ReverseReranker:
    """Puts the originally-last candidate first — proves rerank order is applied."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(i) for i in range(len(texts))]


def test_reranker_reorders_and_truncates(make_service, docs_dir: Path) -> None:
    svc, _ = make_service()
    svc.ingest([docs_dir])
    plain = Retriever(svc.store, fetch_k=10, top_k=10).search("grace period payment")
    reranked = Retriever(svc.store, fetch_k=10, top_k=3, reranker=ReverseReranker()).search(
        "grace period payment"
    )
    assert len(reranked) == 3
    assert reranked[0].text == plain[-1].text
    assert [c.id for c in reranked] == [1, 2, 3]


def test_answer_evaluation_metrics(make_service, docs_dir: Path) -> None:
    def extractive(q, chunks):  # stand-in "LLM": quotes the top fragment
        return GeneratedAnswer(
            answer=chunks[0].text + " [1]", cited_ids=[1], insufficient_context=False
        )

    svc, _ = make_service(grade=lambda q, cs: [1], generate=extractive)
    svc.ingest([docs_dir])
    report = evaluate_answers(svc, load_eval(DATA / "eval.jsonl"))
    assert report.citation_accuracy == 1.0
    assert report.false_refusal_rate == 0.0
    assert report.fact_recall >= 0.7
