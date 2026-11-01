from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_service.config import Settings
from rag_service.embeddings import HashingEmbeddings, HashingSparseEmbeddings
from rag_service.graph import RagDeps
from rag_service.retrieval import Chunk
from rag_service.schemas import GeneratedAnswer
from rag_service.service import RagService

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_path=":memory:",
        embeddings="hashing:1024",
        sparse_embeddings="hashing",
        reranker="none",
        upload_dir=str(tmp_path / "uploads"),
        fetch_k=20,
        top_k=5,
    )


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "docs"
    shutil.copytree(DATA / "docs", target)
    return target


class Recorder:
    """Stub LLM steps that record their inputs."""

    def __init__(self) -> None:
        self.calls: dict[str, list] = {}

    def log(self, name: str, *args) -> None:
        self.calls.setdefault(name, []).append(args)


@pytest.fixture
def make_service(settings: Settings) -> Callable[..., tuple[RagService, Recorder]]:
    def factory(
        *,
        grade: Callable[[str, list[Chunk]], list[int]] | None = None,
        generate: Callable[[str, list[Chunk]], GeneratedAnswer] | None = None,
        rewrite: Callable[[str, str], str] | None = None,
        contextualize: Callable[[str, list], str] | None = None,
        **overrides,
    ) -> tuple[RagService, Recorder]:
        s = settings.model_copy(update=overrides)
        rec = Recorder()
        holder: dict[str, RagService] = {}

        def _grade(q, chunks):
            rec.log("grade", q, chunks)
            return grade(q, chunks) if grade else [c.id for c in chunks[:3]]

        def _generate(q, chunks):
            rec.log("generate", q, chunks)
            if generate:
                return generate(q, chunks)
            return GeneratedAnswer(answer="Answer [1].", cited_ids=[1], insufficient_context=False)

        def _rewrite(q, failed):
            rec.log("rewrite", q, failed)
            return rewrite(q, failed) if rewrite else q + " (rephrased)"

        def _contextualize(q, history):
            rec.log("contextualize", q, history)
            return contextualize(q, history) if contextualize else q

        def _retrieve(query):
            rec.log("retrieve", query)
            return holder["svc"].retriever.search(query)

        deps = RagDeps(_retrieve, _contextualize, _grade, _rewrite, _generate)
        svc = RagService.from_settings(
            s, dense=HashingEmbeddings(1024), sparse=HashingSparseEmbeddings(), deps=deps
        )
        holder["svc"] = svc
        return svc, rec

    return factory
