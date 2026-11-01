"""Qdrant-backed document store.

The collection is created explicitly (not via `from_documents`) to control:
named vectors, cosine distance, the IDF modifier for sparse vectors (BM25 needs
corpus-level IDF, which Qdrant computes) and a payload index on `metadata.source`
for fast per-file deletes during re-indexing.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models

from rag_service.config import Settings

log = logging.getLogger(__name__)

DENSE, SPARSE = "dense", "sparse"
SOURCE_KEY = "metadata.source"


def make_client(s: Settings) -> QdrantClient:
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    if s.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=s.qdrant_path)  # embedded, persisted to disk, no server


class DocStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: Embeddings,
        sparse: SparseEmbeddings | None = None,
        *,
        server: bool = False,
    ) -> None:
        self.client, self.collection, self.server = client, collection, server
        self.hybrid = sparse is not None
        self._ensure_collection(dense)
        self.vs = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID if self.hybrid else RetrievalMode.DENSE,
            vector_name=DENSE,
            sparse_vector_name=SPARSE,
        )

    def _ensure_collection(self, dense: Embeddings) -> None:
        if self.client.collection_exists(self.collection):
            return
        dim = len(dense.embed_query("dimension probe"))
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config=(
                {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                if self.hybrid
                else None
            ),
        )
        if self.server:  # payload indexes only matter on a real server (embedded mode ignores them)
            self.client.create_payload_index(
                self.collection, SOURCE_KEY, models.PayloadSchemaType.KEYWORD
            )
        log.info("created collection %s (dim=%d, hybrid=%s)", self.collection, dim, self.hybrid)

    @staticmethod
    def _source_filter(source: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key=SOURCE_KEY, match=models.MatchValue(value=source))]
        )

    def stored_hash(self, source: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=self._source_filter(source), limit=1, with_payload=True
        )
        return points[0].payload["metadata"].get("doc_hash") if points else None

    def delete_source(self, source: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source)),
        )

    def add(self, chunks: list[Document], ids: list[str]) -> None:
        self.vs.add_documents(chunks, ids=ids)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.vs.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count


def make_store(s: Settings, dense: Embeddings, sparse: SparseEmbeddings | None) -> DocStore:
    return DocStore(make_client(s), s.collection, dense, sparse, server=bool(s.qdrant_url))
