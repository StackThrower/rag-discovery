"""Dense and sparse embedding factories.

Hybrid search = dense vectors (semantics, synonyms, other languages) + sparse BM25-style
vectors (exact terms: product codes, error ids, names). Qdrant fuses both rankings with
Reciprocal Rank Fusion: https://qdrant.tech/articles/hybrid-search/

`hashing` implementations need no model download — used for offline tests/demos.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, SparseEmbeddings, SparseVector

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


def _bucket(token: str, size: int) -> int:
    # stable across processes (unlike built-in hash()) — required for persisted vectors
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % size


class HashingEmbeddings(Embeddings):
    """Lexical 'dense' embedding via the hashing trick (unigrams + bigrams), L2-normalised."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = [0.0] * self.dim
        for tok, cnt in Counter(feats).items():
            vec[_bucket(tok, self.dim)] += 1.0 + math.log(cnt)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashingSparseEmbeddings(SparseEmbeddings):
    """BM25-like term-frequency sparse vectors; IDF is applied by Qdrant (Modifier.IDF)."""

    def __init__(self, k1: float = 1.2, space: int = 2**31 - 1) -> None:
        self.k1, self.space = k1, space

    def _embed(self, text: str) -> SparseVector:
        counts: Counter[int] = Counter(_bucket(t, self.space) for t in tokenize(text))
        idx = sorted(counts)
        # BM25 TF saturation: tf·(k1+1)/(tf+k1) — long repetitive chunks don't dominate
        vals = [counts[i] * (self.k1 + 1) / (counts[i] + self.k1) for i in idx]
        return SparseVector(indices=idx, values=vals)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        # query side: presence only, IDF does the weighting
        idx = sorted({_bucket(t, self.space) for t in tokenize(text)})
        return SparseVector(indices=idx, values=[1.0] * len(idx))


class FastEmbedDense(Embeddings):
    """Local ONNX embeddings via FastEmbed (no torch, CPU-friendly)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def make_dense(spec: str) -> Embeddings:
    provider, _, name = spec.partition(":")
    if provider == "hashing":
        return HashingEmbeddings(int(name or 1024))
    if provider == "fastembed":
        return FastEmbedDense(name)
    from langchain.embeddings import init_embeddings  # openai:text-embedding-3-large, …

    return init_embeddings(spec)


def make_sparse(spec: str) -> SparseEmbeddings | None:
    provider, _, name = spec.partition(":")
    if provider in {"", "none"}:
        return None
    if provider == "hashing":
        return HashingSparseEmbeddings()
    if provider == "fastembed":
        return FastEmbedSparse(model_name=name or "Qdrant/bm25")
    raise ValueError(f"unsupported sparse embeddings: {spec}")
