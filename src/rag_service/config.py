from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # --- LLMs ("provider:model" for init_chat_model) ---
    chat_model: str = "anthropic:claude-sonnet-5-5"
    """Answer generation — quality matters most here."""
    fast_model: str = "anthropic:claude-haiku-5-5"
    """Query contextualization / rewriting / relevance grading — many small calls."""
    temperature: float | None = None

    # --- embeddings / reranking ("provider:model") ---
    embeddings: str = "fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """fastembed:<model> (local ONNX) | openai:<model> | hashing:<dim> (offline, lexical)."""
    sparse_embeddings: str = "fastembed:Qdrant/bm25"
    """Sparse side of hybrid search: fastembed:Qdrant/bm25 | hashing | none (dense only)."""
    reranker: str = "none"
    """none | fastembed:<cross-encoder>, e.g. fastembed:jinaai/jina-reranker-v2-base-multilingual"""

    # --- Qdrant ---
    qdrant_url: str | None = None
    """http://localhost:6333 for a server; unset → embedded local storage at qdrant_path."""
    qdrant_path: str = ".qdrant"
    """Embedded mode directory, or ':memory:'."""
    qdrant_api_key: str | None = None
    collection: str = "docs"

    pdf_ocr: bool = False
    """OCR for scanned PDFs inside pymupdf4llm (requires Tesseract)."""

    # --- chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- retrieval ---
    fetch_k: int = 20
    """Candidates from hybrid search before reranking / grading."""
    top_k: int = 5
    """Chunks that reach the generator."""
    grade_documents: bool = True
    max_query_rewrites: int = 1

    upload_dir: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
