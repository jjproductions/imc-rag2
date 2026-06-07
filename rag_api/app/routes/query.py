import time
import logging
from fastapi import APIRouter
from app.models.schemas import QueryRequest, AnswerResponse, RetrievedChunk
from app.services.retriever import search_similar
from app.services.prompt import build_messages
from app.services.llm import get_llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["query"])

@router.post("/query", response_model=AnswerResponse)
async def query(req: QueryRequest) -> AnswerResponse:
    t0 = time.time()
    top_k = req.top_k or settings.TOP_K
    try:
        chunks = search_similar(req.question, top_k=top_k)
    except Exception as e:
        logger.error(f"Error during similar search: {e}", exc_info=True)
        usage = {
            "top_k": top_k,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        fallback_msg = "I am sorry, but the document database is temporarily unavailable. Please try again later."
        return AnswerResponse(answer=fallback_msg, sources=[], usage=usage)
    
    messages = build_messages(req.question, chunks)
    
    client = get_llm_client()

    resp = await client.chat_once(
        model=settings.ACTIVE_LLM_MODEL,
        messages=messages,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )
    content = resp.get("message", {}).get("content", "")
    usage = {
        "top_k": top_k,
        "latency_ms": int((time.time() - t0) * 1000),
    }
    logger.info(f"Query answered in {usage['latency_ms']} ms using top_k={top_k}")
    sources = []
    normalized_content = content.lower()
    if "don't know" not in normalized_content and "do not know" not in normalized_content:
        for c in chunks:
            # Be defensive: skip results that don't include required fields
            source_id = c.get("source_id") if isinstance(c, dict) else None
            chunk_id = c.get("chunk_id") if isinstance(c, dict) else None
            text = c.get("text") if isinstance(c, dict) else None
            if source_id is None or chunk_id is None or text is None:
                continue
            sources.append(RetrievedChunk(
                source_id=source_id,
                chunk_id=chunk_id,
                text=text,
                source_path=c.get("source_path", ""),
                page=c.get("page"),
                score=c.get("score"),
                section=c.get("section_path"),
            ))
    return AnswerResponse(answer=content, sources=sources, usage=usage)