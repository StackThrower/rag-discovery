"""Two-stage retrieval: wide hybrid recall (fetch_k) → precise cross-encoder rerank (top_k).

Bi-encoder embeddings compare query and chunk *independently* — fast, good recall.
A cross-encoder reads query and chunk *together* — slow, much better precision.
So: retrieve ~20 cheaply, rerank them, keep ~5 for the LLM.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from rag_service.store import DocStore


class Chunk(BaseModel):
    """What travels through the graph (serializable, provider-agnostic)."""

    id: int
    text: str
    source: str
    file_name: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    score: float
    rerank_score: float | None = None

    def location(self) -> str:
        parts = [self.file_name]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Local ONNX cross-encoder (e.g. jinaai/jina-reranker-v2-base-multilingual)."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.rerank(query, texts)]


def make_reranker(spec: str) -> Reranker | None:
    provider, _, name = spec.partition(":")
    if provider in {"", "none"}:
        return None
    if provider == "fastembed":
        return FastEmbedReranker(name)
    raise ValueError(f"unsupported reranker: {spec}")


class Retriever:
    def __init__(
        self, store: DocStore, *, fetch_k: int, top_k: int, reranker: Reranker | None = None
    ) -> None:
        self.store, self.fetch_k, self.top_k, self.reranker = store, fetch_k, top_k, reranker

    def search(self, query: str) -> list[Chunk]:
        hits = self.store.search(query, k=self.fetch_k)
        chunks = [
            Chunk(
                id=i,
                text=doc.page_content,
                source=doc.metadata.get("source", "?"),
                file_name=doc.metadata.get("file_name", "?"),
                title=doc.metadata.get("title"),
                section=doc.metadata.get("section") or None,
                page=doc.metadata.get("page"),
                score=float(score),
            )
            for i, (doc, score) in enumerate(hits, 1)
        ]
        if self.reranker and chunks:
            scores = self.reranker.score(query, [c.text for c in chunks])
            for c, sc in zip(chunks, scores, strict=True):
                c.rerank_score = sc
            chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        top = chunks[: self.top_k]
        return [c.model_copy(update={"id": i}) for i, c in enumerate(top, 1)]
