import time
from fastapi import APIRouter
from app.models.schemas import QueryRequest, AnswerResponse, RetrievedChunk
from app.services.retriever import search_similar
from app.services.prompt import build_messages
from app.services.llm import get_ollama
from app.core.config import settings

router = APIRouter(prefix="", tags=["query"])

@router.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest):
    t0 = time.time()
    top_k = req.top_k or settings.TOP_K
    chunks = search_similar(req.question, top_k=top_k)
    messages = build_messages(req.question, chunks)
    client = get_ollama()
    resp = client.chat_once(
        model=settings.OLLAMA_MODEL,
        messages=messages,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )
    content = resp.get("message", {}).get("content", "")
    usage = {
        "top_k": top_k,
        "latency_ms": int((time.time() - t0) * 1000),
    }
    sources = [RetrievedChunk(**{
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "source_path": c["source_path"],
        "page": c.get("page"),
        "score": c.get("score"),
    }) for c in chunks]
    return AnswerResponse(answer=content, sources=sources, usage=usage)