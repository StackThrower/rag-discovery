"""Structure-aware chunking with contextual headers.

1. Split by Markdown headings first, so a chunk never mixes two sections.
2. Then split long sections recursively (paragraph → line → sentence → word).
3. Prefix every chunk with "document title › section path". A chunk like
   "Response time is 4 hours" is useless alone; "[Support policy › Priorities › P2]"
   makes it retrievable and citable.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    by_header = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        for section in by_header.split_text(doc.page_content):
            meta = {**doc.metadata, **section.metadata}
            path = " › ".join(meta[h] for h in ("h1", "h2", "h3") if h in meta)
            if path.startswith(meta.get("title", "")):
                path = path[len(meta.get("title", "")) :].lstrip(" ›")
            header = " › ".join(x for x in (meta.get("title"), path) if x)
            for piece in by_size.split_text(section.page_content):
                if not piece.strip():
                    continue
                chunks.append(
                    Document(
                        page_content=f"[{header}]\n{piece}",
                        metadata={k: v for k, v in meta.items() if k not in {"h1", "h2", "h3"}}
                        | {"section": path},
                    )
                )
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks
