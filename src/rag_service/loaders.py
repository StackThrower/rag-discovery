"""File loaders → LangChain `Document`s with provenance metadata (source, page, title)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            yield p


def _title(text: str, fallback: str) -> str:
    m = _H1.search(text)
    return m.group(1).strip() if m else fallback


def load_file(path: Path, source: str, *, pdf_ocr: bool = False) -> list[Document]:
    base = {"source": source, "file_name": path.name}
    if path.suffix.lower() == ".pdf":
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(
            str(path),
            page_chunks=True,
            header=False,
            footer=False,
            show_progress=False,
            use_ocr=pdf_ocr,  # scanned PDFs: run them through 02-pdf-extractor or enable OCR
        )
        title = _title(pages[0]["text"], path.stem) if pages else path.stem
        return [
            Document(
                page_content=p["text"],
                metadata={**base, "title": title, "page": p["metadata"].get("page_number", i + 1)},
            )
            for i, p in enumerate(pages)
            if p["text"].strip()
        ]
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={**base, "title": _title(text, path.stem)})]
