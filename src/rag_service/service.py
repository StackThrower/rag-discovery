"""Facade wiring settings → embeddings → Qdrant → retriever → graph. Used by CLI and API."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings

from rag_service.config import Settings, get_settings
from rag_service.embeddings import make_dense, make_sparse
from rag_service.graph import RagDeps, build_rag_graph, llm_deps, make_checkpointer
from rag_service.ingest import IngestReport, ingest
from rag_service.llm import init_llm
from rag_service.retrieval import Reranker, Retriever, make_reranker
from rag_service.schemas import Answer
from rag_service.store import DocStore, make_store


class RagService:
    def __init__(
        self,
        settings: Settings,
        store: DocStore,
        retriever: Retriever,
        deps: RagDeps | None = None,
    ) -> None:
        self.s, self.store, self.retriever = settings, store, retriever
        if deps is None:
            deps = llm_deps(
                retriever.search,
                init_llm(settings.chat_model, settings.temperature),
                init_llm(settings.fast_model, settings.temperature),
            )
        self.graph = build_rag_graph(
            deps,
            grade_documents=settings.grade_documents,
            max_rewrites=settings.max_query_rewrites,
            checkpointer=make_checkpointer(),
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        dense: Embeddings | None = None,
        sparse: SparseEmbeddings | None = None,
        reranker: Reranker | None = None,
        deps: RagDeps | None = None,
    ) -> RagService:
        s = settings or get_settings()
        dense = dense or make_dense(s.embeddings)
        sparse = sparse if sparse is not None else make_sparse(s.sparse_embeddings)
        store = make_store(s, dense, sparse)
        retriever = Retriever(
            store,
            fetch_k=s.fetch_k,
            top_k=s.top_k,
            reranker=reranker if reranker is not None else make_reranker(s.reranker),
        )
        return cls(s, store, retriever, deps)

    def ingest(self, paths: Iterable[Path]) -> IngestReport:
        return ingest(paths, self.store, self.s)

    def ask(self, question: str, thread_id: str | None = None) -> Answer:
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        return self.graph.invoke({"question": question}, config)["answer"]
