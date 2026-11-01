"""HTTP API: `rag serve` or `uvicorn rag_service.api:app` → http://127.0.0.1:8000/docs"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag_service.loaders import SUPPORTED
from rag_service.schemas import Answer
from rag_service.service import RagService

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    thread_id: str | None = Field(None, description="Reuse to keep conversation context")


class IngestResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]
    failed: dict[str, str]
    chunks: int


@lru_cache
def get_service() -> RagService:
    return RagService.from_settings()


Service = Annotated[RagService, Depends(get_service)]


def _safe_name(name: str) -> str:
    # never trust client file names: no paths, no traversal
    return re.sub(r"[^\w.\-]", "_", Path(name).name)[:120] or "upload"


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats(svc: Service) -> dict[str, int | str]:
        return {"collection": svc.s.collection, "chunks": svc.store.count()}

    @app.post("/ask", response_model=Answer)
    async def ask(req: AskRequest, svc: Service) -> Answer:
        return await run_in_threadpool(svc.ask, req.question, req.thread_id)

    @app.post("/documents", response_model=IngestResponse)
    async def upload(svc: Service, files: Annotated[list[UploadFile], File()]) -> IngestResponse:
        target = Path(svc.s.upload_dir)
        target.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            name = _safe_name(f.filename or "upload")
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(415, f"{name}: supported types are {sorted(SUPPORTED)}")
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name}: file too large")
            (target / name).write_bytes(data)
            saved.append(target / name)
        report = await run_in_threadpool(svc.ingest, saved)
        return IngestResponse(**report.__dict__)

    return app


app = create_app()
