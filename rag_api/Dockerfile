FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1             PYTHONUNBUFFERED=1             TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}             HF_HOME=${HF_HOME:-/models}             PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends             curl             && rm -rf /var/lib/apt/lists/*

# Python deps (minimal & offline-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_api/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]