FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1             PYTHONUNBUFFERED=1             TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}             HF_HOME=${HF_HOME:-/models}             PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends             curl             && rm -rf /var/lib/apt/lists/*

# Python deps (minimal & offline-friendly)
RUN pip install --no-cache-dir             fastapi==0.115.5             uvicorn[standard]==0.30.6             python-multipart==0.0.9             sentence-transformers==3.0.1             qdrant-client==1.9.1             pydantic==2.9.2             pydantic-settings==2.6.1             sse-starlette==2.2.1             PyPDF2==3.0.1             httpx==0.27.2             markdown==3.6             orjson==3.10.7             pytest==8.3.3

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]