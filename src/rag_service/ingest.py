"""Idempotent ingestion: re-running on the same folder does no work for unchanged files.

* file hash stored with every chunk → unchanged file = skip (no embedding cost);
* changed file = delete its old chunks, insert new ones (no stale duplicates);
* chunk ids are deterministic UUIDv5 → upserts are safe to retry.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rag_service.chunking import chunk_documents
from rag_service.config import Settings
from rag_service.loaders import iter_files, load_file
from rag_service.store import DocStore

log = logging.getLogger(__name__)
_NS = uuid.UUID("6f1c5b2e-3f0e-4c8e-9a51-1d2f7e0b9c44")


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0


def _source_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ingest(paths: Iterable[Path], store: DocStore, s: Settings) -> IngestReport:
    report = IngestReport()
    for path in iter_files(paths):
        source = _source_id(path)
        try:
            doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            if store.stored_hash(source) == doc_hash:
                report.skipped.append(source)
                continue
            docs = load_file(path, source, pdf_ocr=s.pdf_ocr)
            chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
            for c in chunks:
                c.metadata["doc_hash"] = doc_hash
            ids = [
                str(uuid.uuid5(_NS, f"{source}#{c.metadata['chunk_index']}#{doc_hash}"))
                for c in chunks
            ]
            store.delete_source(source)
            if chunks:
                store.add(chunks, ids)
            report.indexed.append(source)
            report.chunks += len(chunks)
            log.info("indexed %s: %d chunks", source, len(chunks))
        except Exception as exc:  # one bad file must not stop the batch
            log.exception("failed to ingest %s", source)
            report.failed[source] = f"{type(exc).__name__}: {exc}"
    return report
