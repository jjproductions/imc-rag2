import json, time, uuid, asyncio, anyio, re
from typing import AsyncGenerator, Dict, Any, Iterable
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import StreamRequest, OpenAIChatCompletionRequest
from app.services.retriever import search_similar
from app.services.prompt import build_messages
from app.services.llm import get_ollama
from app.core.config import settings

router = APIRouter(prefix="", tags=["stream"])


@router.post("/stream")
async def stream(req: StreamRequest):
    top_k = req.top_k or settings.TOP_K
    trace_id = req.trace_id or str(uuid.uuid4())
    chunks = search_similar(req.question, top_k=top_k)
    messages = build_messages(req.question, chunks)
    client = get_ollama()

    start = time.time()
    total_ms = 0

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        # Stream from Ollama and forward as SSE
        # Each 'data' is JSON: {"delta":"..."}; final contains usage
        async with asyncio.Semaphore(1):
            for ev in client.chat_stream(
                model=settings.OLLAMA_MODEL,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            ):
                delta = ev.get("message", {}).get("content", "")
                if delta:
                    yield {
                        "event": "token",
                        "data": json.dumps({"delta": delta, "trace_id": trace_id}),
                    }
            total_ms = int((time.time() - start) * 1000)
            usage = {"top_k": top_k, "latency_ms": total_ms}
            yield {
                "event": "complete",
                "data": json.dumps(
                    {"complete": True, "usage": usage, "trace_id": trace_id}
                ),
            }

    print(
        f"Starting SSE stream for trace_id={trace_id} with top_k={top_k} completed in {total_ms} ms"
    )
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ------------------ OpenAI-compatible endpoint ------------------

openai_router = APIRouter(tags=["openai-compat"])


@openai_router.post("/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest, request: Request):
    """
    OpenAI-compatible, but RAG-enabled:
    - Takes last user message as the question for retrieval.
    - Augments context and streams model output in OpenAI delta format when stream=True.
    """
    # Extract last user message
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    top_k = settings.TOP_K
    chunks = search_similar(last_user, top_k=top_k)
    messages = build_messages(last_user, chunks)

    client = get_ollama()  # your adapter (with async chat_stream/chat_once)
    model = req.model or settings.OLLAMA_MODEL
    temperature = req.temperature or settings.TEMPERATURE
    max_tokens = req.max_tokens or settings.MAX_TOKENS

    created = int(time.time())
    comp_id = f"chatcmpl-{uuid.uuid4().hex}"
    print(
        f"OpenAI chat/completions request id={comp_id}, model={model}, stream={req.stream}"
    )

    # ── STREAMING ────────────────────────────────────────────────────────────────
    if req.stream:
        print("Processing streaming OpenAI chat completion request")
        start = time.time()
        total_ms = 0

        async def gen():
            # Initial role delta (optional but common)
            first_chunk = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            # SSE events must end with a blank line
            yield f"data: {json.dumps(first_chunk)}\n\n"
            # Cooperative yield to flush immediately
            await anyio.sleep(0)

            # Stream upstream deltas as they arrive (no buffering!)
            async for ev in client.chat_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                delta = ev.get("message", {}).get("content", "")
                if not delta:
                    continue

                data = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
                # Help some ASGI stacks flush per token
                await anyio.sleep(0)

            # Final finish chunk + DONE
            elapsed_s = round(time.time() - start, 3)

            # Extract nearby/article numbers from the retrieved chunks so the UI can show them
            source_articles = {}
            try:
                for c in chunks:
                    sid = c.get("source_id") or c.get("source_path")
                    text = c.get("section_path", "")
                    if not sid or not text:
                        continue
                    # Look for common article patterns (e.g. 'Article 5' or 'Article V')
                    m_arabic = re.search(r"Article\s+(\d+)", text, re.IGNORECASE)
                    m_roman = re.search(r"Article\s+([IVXLCDM]+)", text, re.IGNORECASE)
                    found = None
                    if m_arabic:
                        found = f"Article {m_arabic.group(1)}"
                    elif m_roman:
                        found = f"Article {m_roman.group(1).upper()}"
                    if found:
                        source_articles.setdefault(sid, set()).add(found)
            except Exception:
                source_articles = {}

            # Normalize sets to lists
            source_articles = (
                {k: list(v) for k, v in source_articles.items()}
                if source_articles
                else {}
            )

            done = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"time_s": elapsed_s},
                "time_taken_s": elapsed_s,
                "time_taken": f"Time Taken: {elapsed_s}s",
                "source_articles": source_articles,
            }
            print(
                f"Completed streaming response for id={comp_id} in {elapsed_s} seconds"
            )
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # important if any nginx is in the path
            },
        )

    # ── NON-STREAMING ───────────────────────────────────────────────────────────
    print("Processing non-streaming OpenAI chat completion request")
    resp = await client.chat_once(  # async call for consistency
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.get("message", {}).get("content", "")
    data = {
        "id": comp_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(content=data)


@openai_router.get("/models")
async def openai_models():
    """Return a minimal OpenAI-compatible model list so frontends can enumerate models."""
    model_id = settings.OLLAMA_MODEL
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
            }
        ],
    }
