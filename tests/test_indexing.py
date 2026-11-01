from __future__ import annotations

from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from rag_service.chunking import chunk_documents
from rag_service.loaders import load_file


def test_chunks_carry_contextual_header_and_section() -> None:
    doc = Document(
        "# Guide\n\n## Billing\n\nInvoices are issued monthly.\n\n## Security\n\nTLS 1.3 only.",
        metadata={"source": "g.md", "file_name": "g.md", "title": "Guide"},
    )
    chunks = chunk_documents([doc], chunk_size=500, overlap=50)
    assert [c.metadata["section"] for c in chunks] == ["Billing", "Security"]
    assert chunks[0].page_content.startswith("[Guide › Billing]\n")
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]


def test_long_sections_are_split_with_overlap() -> None:
    body = " ".join(f"Sentence number {i} about quotas." for i in range(200))
    doc = Document(f"# T\n\n## S\n\n{body}", metadata={"title": "T", "file_name": "t.md"})
    chunks = chunk_documents([doc], chunk_size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c.page_content) <= 300 + len("[T › S]\n") for c in chunks)


def test_pdf_loader_keeps_page_numbers(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    for i in range(2):
        page = pdf.new_page()
        # realistic body text: a lone line at the very top would be dropped as a running header
        page.insert_htmlbox(
            page.rect + (50, 50, -50, -50),
            f"<h1>Quotas {i + 1}</h1><p>The storage quota is {i + 1}00 GB per project.</p>",
        )
    path = tmp_path / "quota.pdf"
    pdf.save(path)
    docs = load_file(path, "quota.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "200 GB" in docs[1].page_content


def test_ingest_is_idempotent_and_replaces_changed_files(make_service, docs_dir: Path) -> None:
    svc, _ = make_service()
    first = svc.ingest([docs_dir])
    assert len(first.indexed) == 6 and not first.failed
    total = svc.store.count()

    second = svc.ingest([docs_dir])
    assert second.indexed == [] and len(second.skipped) == 6
    assert svc.store.count() == total  # no duplicates

    sla = docs_dir / "sla.md"
    sla.write_text("# SLA\n\n## Uptime\n\nUptime is 99.99% for everyone.\n", encoding="utf-8")
    third = svc.ingest([docs_dir])
    assert len(third.indexed) == 1 and third.indexed[0].endswith("sla.md")
    hits = svc.retriever.search("uptime commitment Enterprise 99.95")
    sla_texts = [h.text for h in hits if h.file_name == "sla.md"]
    assert sla_texts and all("99.95" not in t for t in sla_texts)  # old chunks are gone


def test_bad_file_does_not_stop_batch(make_service, docs_dir: Path) -> None:
    (docs_dir / "broken.pdf").write_bytes(b"%PDF-1.7 garbage")
    report = make_service().__getitem__(0).ingest([docs_dir])
    assert len(report.indexed) == 6
    assert any(k.endswith("broken.pdf") for k in report.failed)
