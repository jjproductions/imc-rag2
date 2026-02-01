# Local RAG Chat (Qdrant + bge-m3 + FastAPI + Ollama + OpenWebUI)

A fully local Retrieval-Augmented Generation (RAG) system:

- **Qdrant** for vector search
- **BAAI/bge-m3** for embeddings (1024-dim, normalized, cosine)
- **FastAPI** backend (`rag_api`) split into **Ingestion** & **Query/Stream** routes
- **Ollama** as local LLM runtime
- **OpenWebUI** frontend pointing to the backend's **OpenAI-compatible** `/v1` endpoint
- **Streaming** (SSE and OpenAI-style)
- **No cloud dependencies at runtime**, with an **air-gapped** mode

---

## Architecture

```
+------------------+        +------------------+        +-----------------------+
|  OpenWebUI       | <----> |   RAG API        | <----> |  Ollama (Local LLM)   |
| (Frontend)       |  /v1   | FastAPI          |        |  llama3.1:*           |
|                  |        |  - /ingest       |        +-----------------------+
|                  |        |  - /query        |
|                  |        |  - /stream (SSE) |        +-----------------------+
|                  |        |  - /v1/chat/...  | <----> |  Qdrant (Vector DB)   |
+------------------+        +------------------+        +-----------------------+

Ingestion: local docs -> chunk -> bge-m3 -> Qdrant
Query: question -> retrieve top-k -> prompt -> Ollama -> stream tokens -> UI
```

---

## Quick Start

1. **Clone and prepare**
   ```bash
   cp .env.example .env
   # Optionally tweak values (API_KEY, OLLAMA_MODEL, etc.)
   ```

2. **Start the stack**
   ```bash
   make up
   ```

3. **(Online) Pull the default Ollama model**
   ```bash
   make pull-model
   ```

4. **Ingest sample docs**
   ```bash
   make ingest path=./docs
   ```

5. **Ask a question (non-streaming)**
   ```bash
   make query q="What is IMC?"
   ```

6. **Stream a response (SSE)**
   ```bash
   make stream q="What is IMC?"
   ```

7. **Open the UI**
   - Visit `http://localhost:3000`
   - OpenWebUI is already configured to call the RAG API’s OpenAI-compatible endpoint at `http://rag-api:8000/v1` with `OPENAI_API_KEY` from `.env`.

---

## Endpoints

- `POST /ingest` — body `{"path": "/absolute/or/mounted/path"}`  
- `POST /query` — body `{"question":"...", "top_k":5}`  
- `POST /stream` — body `{"question":"...", "top_k":5}` (SSE with `data: {"delta": "..."}`)  
- `POST /v1/chat/completions` — OpenAI-compatible; supports `stream: true`  

> **Auth:** All endpoints require `Authorization: Bearer <API_KEY>`.

---

## Offline / Air‑Gapped Setup

To run fully offline:

1. On a connected machine, **pre-download** the models:
   - **Sentence Transformers cache** for `BAAI/bge-m3`
   - **Ollama model** (e.g., `llama3.1:8b-instruct-q4_0`)

2. Copy/carry the following directories into your air-gapped host:
   - `./models` (Hugging Face cache location for embeddings)
   - `~/.ollama` (or export from a similar path) to an external drive

3. Start with **offline env flags**:
   ```env
   TRANSFORMERS_OFFLINE=1
   HF_HOME=/models
   ```
   Ensure `./models` is mounted to the `rag-api` container (already set in `docker-compose.yml`):
   ```yaml
   rag-api:
     volumes:
       - ./models:/models
   ```
   And ensure the **Ollama** container has your pre-seeded models under its default volume `ollama_models` (already defined). If you imported them manually, place files under `./ollama_models` and mount accordingly.

4. Bring up the stack:
   ```bash
   make up
   ```

5. **No external calls** will be made:
   - `TRANSFORMERS_OFFLINE=1` prevents HF downloads.
   - Models are loaded from `/models` (mounted).
   - LLM runs from local Ollama cache.

---

## Swap LLM or Embedding Model

- **LLM**: Edit `.env`
  ```env
  OLLAMA_MODEL=llama3.1:8b-instruct-q4_0
  TEMPERATURE=0.2
  MAX_TOKENS=1024
  ```
  Then `make restart` and (if online) `make pull-model`.

- **Embeddings**: Edit `.env`
  ```env
  EMBEDDING_MODEL=BAAI/bge-m3
  ```
  Rebuild `rag-api`:
  ```bash
  make up
  ```
  *Note:* Qdrant collection is fixed to 1024 dims for `bge-m3`. If you change to a model with different dimension, **create a new collection** or drop and recreate with the correct size.

---

## cURL Examples

- **Ingest a folder**
  ```bash
  curl -fsS -H "Authorization: Bearer $API_KEY"                -H "Content-Type: application/json"                -d "{"path":"$PWD/docs"}"                http://localhost:8000/ingest
  ```

- **Query (non-streaming)**
  ```bash
  curl -fsS -H "Authorization: Bearer $API_KEY"                -H "Content-Type: application/json"                -d '{"question":"What is IMC?"}'                http://localhost:8000/query | jq
  ```

- **Stream (SSE)**
  ```bash
  curl -N -H "Authorization: Bearer $API_KEY"                -H "Content-Type: application/json"                -d '{"question":"What is IMC?"}'                http://localhost:8000/stream
  ```

- **OpenAI-compatible streaming**
  ```bash
  curl -N -H "Authorization: Bearer $API_KEY"                -H "Content-Type: application/json"                -d '{
          "model": "llama3.1:8b-instruct-q4_0",
          "messages":[{"role":"user","content":"What is IMC?"}],
          "stream": true
       }'                http://localhost:8000/v1/chat/completions
  ```

---

## Minimal Smoke Test Transcript

```text
$ make up
... containers start, healthchecks pass ...

$ make ingest path=./docs
{"inserted": 2, "skipped": 0}

$ make query q="What is IMC?"
{
  "answer": "The Institute of Music for Children (IMC) empowers youth through arts education (source: docs/policy_snippet.txt#0).",
  "sources": [
    {
      "doc_id": "docs/policy_snippet.txt",
      "chunk_id": 0,
      "text": "The Institute of Music for Children (IMC) empowers youth through arts education...",
      "source_path": "/workspace/docs/policy_snippet.txt",
      "page": null,
      "score": 0.88
    }
  ],
  "usage": {"top_k": 5, "latency_ms": 450}
}

$ make stream q="What is IMC?"
event: token
data: {"delta":"The Institute of Music for Children (IMC) empowers youth through arts education ","trace_id":"..."}
event: token
data: {"delta":"(source: docs/policy_snippet.txt#0).","trace_id":"..."}
event: complete
data: {"complete": true, "usage": {"top_k":5, "latency_ms": 620}, "trace_id": "..."}
```

---

**Security**: This is a local dev stack. The API requires a bearer token; do not expose ports publicly.
**Determinism**: Default `TEMPERATURE=0.2`. Adjust as needed.
**Collections**: If you re-embed with a different embedding size, create a new Qdrant collection configured accordingly.