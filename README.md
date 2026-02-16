# Local RAG Chat (Qdrant + bge-m3 + FastAPI + Ollama + OpenWebUI)

A fully local Retrieval-Augmented Generation (RAG) system:

- **Qdrant** for vector search (persisted locally)
- **BAAI/bge-m3** for embeddings (1024-dim, normalized, cosine)
- **FastAPI** backend (`rag_api`) split into **Query/Stream** routes
- **Ollama** as local LLM runtime (supports Llama 3, Mistral, etc.)
- **OpenWebUI** frontend pointing to the backend's **OpenAI-compatible** `/v1` endpoint
- **Citation Linking**: Generates links to specific pages in your documents if `DOC_BASE_URL` is configured.
- **Streaming** (SSE and OpenAI-style)
- **No cloud dependencies at runtime**, with an **air-gapped** mode

---

## Architecture

```
+------------------+        +------------------+        +-----------------------+
|  OpenWebUI       | <----> |   RAG API        | <----> |  Ollama (Local LLM)   |
| (Frontend)       |  /v1   | FastAPI          |        |  mistral-small3.2:*   |
|                  |        |  - /query        |        +-----------------------+
|                  |        |  - /stream (SSE) |        +-----------------------+
|                  |        |  - /v1/chat/...  | <----> |  Qdrant (Vector DB)   |
+------------------+        +------------------+        +-----------------------+

Ingestion: Handled by external `board-ingest-api` service.
Query: question -> retrieve top-k -> prompt -> Ollama -> stream tokens -> UI
```

---

## Quick Start

1. **Clone and prepare**

   ```bash
   cp .env.example .env
   # Tweaking .env:
   # - API_KEY: Your secret key
   # - QDRANT_COLLECTION: Name of your collection (e.g., board-policies)
   # - DOC_BASE_URL: (Optional) Base URL for your documents (e.g. SharePoint) to generate clickable links.
   ```

2. **Start the stack**

   ```bash
   make up
   ```

3. **(Online) Pull the default Ollama model**

   ```bash
   make pull-model
   ```

   *Note: Ensure your `OLLAMA_MODEL` in `.env` matches what you pull.*

4. **Ask a question (CLI)**

   ```bash
   make query q="What is the conflict of interest policy?"
   ```

5. **Stream a response (CLI)**

   ```bash
   make stream q="What is the conflict of interest policy?"
   ```

6. **Open the UI**
   - Visit `http://localhost:3000`
   - OpenWebUI is pre-configured to talk to `http://rag-api:8000/v1` using your `API_KEY`.

---

## Development & Updates

- **Restarting the API**:
  If you modify the Python code in `rag_api/`, simply run:
  ```bash
  make dev
  ```
  This rebuilds and restarts *only* the `rag-api` container, leaving Qdrant and Ollama running.

- **Re-ingestion**:
  If you change the chunking logic or add new files, run `make ingest path=...` again. Content is deduplicated by hash, but improved chunking logic requires a DB reset or overwrite.

---

## Features details

### Page Number Extraction
PDF ingestion uses `PyPDF2` to read files page-by-page. The page number is stored in the vector payload.

### Citation Linking
If `DOC_BASE_URL` is set in `.env`, the system formats citations as:
`[1] [Document Title (Page X)](https://your-base-url/Document.pdf#page=X)`

- White spaces in filenames are automatically URL-encoded.
- `#page=X` is appended for PDF deep linking.

---

## Endpoints

- `POST /query` — body `{"question":"...", "top_k":5}`
- `POST /stream` — body `{"question":"...", "top_k":5}` (SSE)
- `POST /v1/chat/completions` — OpenAI-compatible; supports `stream: true`

> **Auth:** All endpoints require `Authorization: Bearer <API_KEY>`.

---

## Offline / Air‑Gapped Setup

To run fully offline:

1. **Pre-download models**:
   - `BAAI/bge-m3` (HuggingFace cache)
   - Ollama models

2. **Configure `.env`**:
   ```env
   TRANSFORMERS_OFFLINE=1
   HF_HOME=/models
   ```

3. **Mount volumes**:
   Ensure `./models` and `./ollama_models` are populated and mounted (default configuration handles this).

4. **Start**:
   ```bash
   make up
   ```

---

## Security
This is a local dev stack. The API requires a bearer token. Do not expose ports publicly without a reverse proxy (Nginx/Traefik) handling TLS and stricter auth.
