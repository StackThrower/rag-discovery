from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_service import prompts
from rag_service.api import create_app, get_service
from rag_service.schemas import GeneratedAnswer


def test_answer_with_validated_citations(make_service, docs_dir: Path) -> None:
    def generate(q, chunks):
        assert chunks[0].file_name == "incident-runbook.md"
        return GeneratedAnswer(
            answer="Control plane failover [1]. Also see [9].",  # [9] was never provided
            cited_ids=[1, 9],
            insufficient_context=False,
        )

    svc, rec = make_service(grade=lambda q, cs: [1], generate=generate)
    svc.ingest([docs_dir])
    ans = svc.ask("What does error ORB-503 mean?")

    assert [c.n for c in ans.citations] == [1]
    assert ans.citations[0].location.startswith("incident-runbook.md")
    assert "[9]" not in ans.answer and "[1]" in ans.answer
    assert ans.insufficient_context is False
    assert len(rec.calls["generate"][0][1]) == 1  # grading filtered the context


def test_no_relevant_docs_rewrites_once_then_refuses_without_llm(make_service, docs_dir) -> None:
    svc, rec = make_service(grade=lambda q, cs: [], max_query_rewrites=1)
    svc.ingest([docs_dir])
    ans = svc.ask("What is the CEO's favourite pizza?")

    assert ans.insufficient_context and ans.answer == prompts.NO_CONTEXT_ANSWER
    assert len(rec.calls["rewrite"]) == 1
    assert len(rec.calls["retrieve"]) == 2  # original + rewritten query
    assert "generate" not in rec.calls  # refusing costs nothing and can't hallucinate
    assert ans.rewrites == 1


def test_rewrite_recovers_when_second_search_is_relevant(make_service, docs_dir) -> None:
    answers = iter([[], [1]])  # first grading finds nothing, second finds fragment 1
    svc, rec = make_service(
        grade=lambda q, cs: next(answers), rewrite=lambda q, failed: "annual billing discount"
    )
    svc.ingest([docs_dir])
    ans = svc.ask("how do I save money on yearly plans")
    assert rec.calls["retrieve"][1] == ("annual billing discount",)
    assert ans.search_query == "annual billing discount" and not ans.insufficient_context


def test_follow_up_question_uses_history(make_service, docs_dir) -> None:
    def contextualize(q, history):
        assert history[0].content == "What is the uptime commitment for Pro?"
        return "uptime commitment Enterprise plan"

    svc, rec = make_service(contextualize=contextualize)
    svc.ingest([docs_dir])
    svc.ask("What is the uptime commitment for Pro?", thread_id="t1")
    ans = svc.ask("And for Enterprise?", thread_id="t1")

    assert "contextualize" in rec.calls and ans.search_query == "uptime commitment Enterprise plan"
    assert len(rec.calls["contextualize"]) == 1  # first turn has no history → no LLM call


def test_api_upload_and_ask(make_service, tmp_path: Path) -> None:
    svc, _ = make_service()
    app = create_app()
    app.dependency_overrides[get_service] = lambda: svc
    client = TestClient(app)

    doc = b"# Office\n\n## Wi-Fi\n\nThe guest Wi-Fi password rotates every Monday.\n"
    r = client.post("/documents", files=[("files", ("../../evil name.md", doc, "text/markdown"))])
    assert r.status_code == 200, r.text
    assert r.json()["chunks"] == 1
    assert (Path(svc.s.upload_dir) / "evil_name.md").exists()  # traversal neutralised

    bad = client.post("/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415

    ans = client.post("/ask", json={"question": "When does the guest Wi-Fi password change?"})
    assert ans.status_code == 200
    assert ans.json()["citations"][0]["location"].startswith("evil_name.md")
    assert client.get("/stats").json()["chunks"] == 1
