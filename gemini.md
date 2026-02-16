# Gemini CLI — Quick Setup & Usage

Purpose

Prerequisites (macOS)


Repository integration notes

Next steps

If you'd like, I can also add specific example commands showing how to call the Gemini REST or Python SDK from `rag_api/app/services`.

Repository overview

- Repository: imc-rag2 (root)
- Purpose: A small Retrieval-Augmented Generation (RAG) API that ingests data, stores vectors in Qdrant, retrieves context, and formats prompts for LLM responses.
- Key files and folders:
  - `rag_api/app/main.py` - ASGI entrypoint (runs under `uvicorn`).
  - `rag_api/app/services/llm.py` - LLM client wrappers (primary place to swap in Gemini SDK or client).
  - `rag_api/app/services/embeddings.py` - Embedding generation logic.
  - `rag_api/app/services/qdrant_client.py` - Qdrant vector DB client and helpers.
  - `rag_api/app/services/retriever.py` - Retrieval and ranking logic.
  - `rag_api/app/services/prompt.py` - Prompt templates and formatting.
  - `rag_api/app/routes/ingest.py` - Ingestion endpoints (adds documents/vectors).
  - `rag_api/app/routes/query.py` - Query endpoint (returns LLM responses using retrieved context).
  - `rag_api/app/routes/stream.py` - Streaming/real-time endpoints.
  - `rag_api/app/models/schemas.py` - Pydantic request/response schemas.
  - `rag_api/app/utils/` - Helpers: `chunking.py`, `caching.py`, `ttlcache.py`, etc.
  - `tests/` - Unit tests: `test_chunking.py`, `test_embeddings.py`, `test_qdrant_roundtrip.py`.
  - `docker-compose.yml`, `Makefile`, `requirements.txt`, `README.md` - repo-level tooling and docs.

How to run locally

- Create and activate a venv, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run the API locally:

```bash
uvicorn rag_api.app.main:app --reload --host 127.0.0.1 --port 8003
```

Gemini integration notes

- To integrate Gemini as the LLM backend, update `rag_api/app/services/llm.py` to call the Gemini Python SDK (or call the REST API). Use `GOOGLE_APPLICATION_CREDENTIALS` or ADC for authentication.
- For quick experiments, use the Gemini CLI (`npx` or `npm`) as documented above.
- For CI or non-interactive usage, prefer service account credentials and environment-driven configuration.

Next steps

- Run `brew install node` then `npm install -g @google/gemini-cli` locally if you want the CLI globally available.
- I can add a concrete example implementation in `rag_api/app/services/llm.py` showing how to call Gemini — tell me if you'd like a sample.

If you'd like, I can also add specific example commands showing how to call the Gemini REST or Python SDK from `rag_api/app/services`.
