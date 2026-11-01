FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# model cache for FastEmbed (ONNX embeddings / reranker) — mounted as a volume in compose
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN useradd --create-home app && mkdir -p /app/data /app/.cache/fastembed && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
