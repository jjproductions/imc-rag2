FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    PIP_NO_CACHE_DIR=1 \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps (minimal & offline-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake models directly into the image to eliminate cold-start latency
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding(model_name='BAAI/bge-large-en-v1.5', cache_dir='/models/fastembed'); SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1', cache_dir='/models/fastembed')"

# Enforce strict offline mode at runtime to eliminate Hugging Face Hub metadata lookups
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY rag_api/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]