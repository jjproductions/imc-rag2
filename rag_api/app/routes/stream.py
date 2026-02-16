import json, time, uuid, asyncio, anyio, re, hashlib, logging
from urllib.parse import quote
from typing import AsyncGenerator, Dict, Any
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import StreamRequest, OpenAIChatCompletionRequest
from app.services.retriever import search_similar
from app.services.prompt import build_messages
from app.services.llm import get_ollama
from app.core.config import settings
from app.utils.caching import (
    extract_final_user_message,
    make_retrieval_cache_key,
    make_cache_key
)
from app.utils.ttlcache import answer_cache, retrieval_cache

logger = logging.getLogger(__name__)

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

    logger.info(
        f"Starting SSE stream for trace_id={trace_id} with top_k={top_k} completed in {total_ms} ms"
    )
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ------------------ OpenAI-compatible endpoint ------------------

openai_router = APIRouter(tags=["openai-compat"])
openai_ws_router = APIRouter(tags=["openai-compat"])


INLINE_SOURCE_RE = re.compile(r"\s*\(source:\s*[^)]+\)", flags=re.IGNORECASE)


def collect_sources(chunks):
    sources = {}
    for c in chunks:
        # Check if 'c' is already the payload (dict) or if it has a .payload attribute
        if hasattr(c, "payload"):
            payload = getattr(c, "payload", {})
            if hasattr(payload, "dict"): # if it's a pydantic model or similar
                 payload = payload.dict()
            elif not isinstance(payload, dict):
                 payload = dict(payload)
        elif isinstance(c, dict):
            # It might be { "payload": {...} } or just { "source_id": ... }
            if "payload" in c and isinstance(c["payload"], dict):
                payload = c["payload"]
            else:
                payload = c
        else:
            payload = {}

        doc_id = payload.get("source_id") or payload.get("doc_id")
        title = payload.get("title") or payload.get("file_name") or doc_id
        
        # 'page' might be a single int, or 'pages' a list [3]
        page = payload.get("page")
        pages_list = payload.get("pages")
        
        if doc_id:
            if doc_id not in sources:
                sources[doc_id] = {"title": title, "pages": set()}
            
            if page is not None:
                sources[doc_id]["pages"].add(page)
            
            if pages_list and isinstance(pages_list, list):
                for p in pages_list:
                    sources[doc_id]["pages"].add(p)
    return sources


def format_sources_block(doc_ids, all_sources):
    # deduplicate while preserving order if possible (though doc_ids input might have dups)
    seen = set()
    unique_ids = []
    for d in doc_ids:
        if d in all_sources and d not in seen:
            unique_ids.append(d)
            seen.add(d)
            
    if not unique_ids:
        return ""
        
    lines = []
    base_url = settings.DOC_BASE_URL

    for i, doc_id in enumerate(unique_ids, start=1):
        meta = all_sources[doc_id]
        title = meta["title"] or doc_id
        
        # Handle "pages" being a set (from collect_sources) or a list/int from raw payload
        raw_pages = meta["pages"]
        if isinstance(raw_pages, set):
             pages = sorted(list(raw_pages))
        elif isinstance(raw_pages, list):
             pages = sorted(raw_pages)
        elif isinstance(raw_pages, int):
             pages = [raw_pages]
        else:
             pages = []

        page_str = ""
        if pages:
            # Format as " (Page 1)" or " (Pages 1, 3)"
            if len(pages) == 1:
                page_str = f" (Page {pages[0]})"
            else:
                page_str = f" (Pages {', '.join(map(str, pages))})"
        
        if base_url:
            # Construct Link
            clean_path = doc_id
            if clean_path.startswith("../"):
                clean_path = clean_path.replace("../", "")
            if clean_path.startswith("./"):
                clean_path = clean_path[2:]
            if clean_path.startswith("data/docs/"): # Common pattern if ingested from root
                clean_path = clean_path.replace("data/docs/", "", 1)
                
            # Encode spaces and special chars, preserve slashes
            clean_path = quote(clean_path, safe='/')
            
            url = f"{base_url.rstrip('/')}/{clean_path.lstrip('/')}"
            if pages and str(url).lower().endswith(".pdf"):
                 url += f"#page={pages[0]}"
            
            lines.append(f"[{i}] [{title}{page_str}]({url})")
        else:
            lines.append(f"[{i}] {title}{page_str}")
        
    return "\n\nSources:\n" + "\n".join(lines)


@openai_ws_router.websocket("/chat/completions")
async def websocket_chat_completions(websocket: WebSocket):
    await websocket.accept()

    # Manual Auth Check
    auth_header = websocket.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("WebSocket missing or invalid authorization header")
        await websocket.close(code=1008)
        return
    
    token = auth_header.split(" ")[1]
    if token != settings.API_KEY:
        logger.warning("WebSocket invalid API key")
        await websocket.close(code=1008)
        return
    
    try:
        # 1) Receive initial request
        data = await websocket.receive_text()
        try:
            req_json = json.loads(data)
            req = OpenAIChatCompletionRequest(**req_json)
        except Exception as e:
            logger.error(f"Invalid WebSocket payload: {e}")
            await websocket.send_text(json.dumps({"error": "Invalid JSON or schema"}))
            await websocket.close(code=1008)
            return

        # 2) Build RAG context (similar to HTTP endpoint)
        start = time.time()
        
        # Safety check for empty messages
        if not req.messages:
             await websocket.send_text(json.dumps({"error": "No messages provided"}))
             await websocket.close()
             return

        parsed_message = extract_final_user_message(
            req.messages[len(req.messages) - 1].content
        )
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"), ""
        )
        top_k = settings.TOP_K

        # Retrieval
        retrieval_key = make_retrieval_cache_key(
            parsed_message,
            top_k,
            getattr(settings, "INDEX_VERSION", "v1"),
        )
        chunks = await retrieval_cache.get(retrieval_key)
        if chunks is None:
            chunks = search_similar(last_user, top_k=top_k)
            await retrieval_cache.set(retrieval_key, chunks)

        messages = build_messages(last_user, chunks)
        
        client = get_ollama()
        model = req.model or settings.OLLAMA_MODEL
        temperature = req.temperature or settings.TEMPERATURE
        max_tokens = req.max_tokens or settings.MAX_TOKENS
        
        comp_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        logger.info(f"WebSocket chat/completions request id={comp_id}, model={model}")

        # 3) Stream Response
        # Initial role chunk
        first_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        await websocket.send_text(json.dumps(first_chunk))
        
        assembled = []
        all_sources = collect_sources(chunks)
        used_doc_ids = list(all_sources.keys())
        
        async for ev in client.chat_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            raw = ev.get("message", {}).get("content", "")
            if not raw:
                continue

            clean = INLINE_SOURCE_RE.sub("", raw)
            if clean:
                assembled.append(clean)
                chunk = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": clean},
                            "finish_reason": None,
                        }
                    ],
                }
                await websocket.send_text(json.dumps(chunk))
                # Yield control to event loop
                await asyncio.sleep(0)

        # Sources Block
        sources_block = format_sources_block(used_doc_ids, all_sources)
        if sources_block:
            chunk = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\n\n" + sources_block},
                        "finish_reason": None,
                    }
                ],
            }
            await websocket.send_text(json.dumps(chunk))

        # Finish
        done_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await websocket.send_text(json.dumps(done_chunk))
        await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected client side")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except:
            pass



@openai_router.post("/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest, request: Request):
    # 1) Build RAG context (cacheable)
    start = time.time()
    total_ms = 0
    parsed_message = extract_final_user_message(
        req.messages[len(req.messages) - 1].content
    )
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    top_k = settings.TOP_K

    logger.debug(
        f"Non WebSocket:Inital user message for OpenAI chat/completions: {json.dumps({'q': parsed_message, 'k': top_k, 'time taken': int((time.time() - start) * 1000)})}"
    )
    # Optional: use retrieval cache
    retrieval_key = make_retrieval_cache_key(
        parsed_message,
        top_k,
        getattr(settings, "INDEX_VERSION", "v1"),
    )
    chunks = await retrieval_cache.get(retrieval_key)
    if chunks is None:
        chunks = search_similar(last_user, top_k=top_k)
        await retrieval_cache.set(retrieval_key, chunks)

    messages = build_messages(last_user, chunks)

    client = get_ollama()
    model = req.model or settings.OLLAMA_MODEL
    temperature = req.temperature or settings.TEMPERATURE
    max_tokens = req.max_tokens or settings.MAX_TOKENS
    index_version = getattr(settings, "INDEX_VERSION", "v1")  # bump when corpus changes

    # 2) Stable cache key for the full request (post-build messages!)
    cache_key = make_cache_key(model, messages, temperature, max_tokens, index_version)

    created = int(time.time())
    comp_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(
        f"OpenAI chat/completions request id={comp_id}, model={model}, stream={req.stream}"
    )

    # 3) STREAMING PATH
    if req.stream:
        logger.debug("Processing streaming OpenAI chat completion request. Time taken: %s", int((time.time() - start) * 1000))

        # If this exact turn already completed (rare but possible on retries),
        # you could serve the cached text as a single delta stream.
        cached = await answer_cache.get(cache_key)

        async def gen():
            # Initial padding to force flush any proxy buffers
            # Many proxies buffer the first 1-4KB. We send 16KB (16384 bytes) of comments.
            yield ": " + (" " * 16384) + "\n\n"
            
            # Initial role chunk
            first_chunk = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            logger.debug("Initial role chunk sent. Time taken: %s", int((time.time() - start) * 1000))
            await anyio.sleep(0)

            if cached is not None:
                # Serve cached answer as one streaming sequence for UI parity
                data = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": cached},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
                done = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Not cached yet: tee the live stream into a buffer
            assembled = []
            all_sources = collect_sources(chunks)
            used_doc_ids = list(
                all_sources.keys()
            )  # simple choice: include all retrieved
            logger.debug("Assembling live stream. Time taken: %s", int((time.time() - start) * 1000))
            async for ev in client.chat_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                raw = ev.get("message", {}).get("content", "")
                if not raw:
                    continue

                clean = INLINE_SOURCE_RE.sub("", raw)
                if clean:
                    assembled.append(clean)
                    data = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": clean},
                                "finish_reason": None,
                            }
                        ],
                    }
                    logger.debug("Live stream chunk: %s ; Time taken: %s", data, int((time.time() - start) * 1000))
                    yield f"data: {json.dumps(data)}\n\n"
                    await anyio.sleep(0)

            # Append final Sources block once
            sources_block = format_sources_block(used_doc_ids, all_sources)
            logger.debug("Sources block: %s ; Time taken: %s", sources_block, int((time.time() - start) * 1000))
            if sources_block:
                assembled.append("\n\n" + sources_block)
                tail = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n\n" + sources_block},
                            "finish_reason": None,
                        }
                    ],
                }
                logger.debug("Sources chunk: %s ; Time taken: %s", tail, int((time.time() - start) * 1000))
                yield f"data: {json.dumps(tail)}\n\n"
                await anyio.sleep(0)

            # Finish
            done = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            logger.debug("Done chunk: %s ; Time taken: %s", done, int((time.time() - start) * 1000))    
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

            total_ms = int((time.time() - start) * 1000)
            usage = {"top_k": top_k, "latency_ms": total_ms}
            logger.info(
                f"OpenAI chat/completions request id={comp_id}, model={model}, stream={req.stream} completed in {total_ms} ms"
            )
            # Cache the full assembled text for the follow-up non-streaming call
            final_text = "".join(assembled)
            await answer_cache.set(cache_key, final_text)

        
        return StreamingResponse(
            gen(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Content-Type-Options": "nosniff",
            },
        )

    # 4) NON‑STREAMING PATH — return cached answer if available
    logger.debug("Processing non-streaming OpenAI chat completion request")
    cached = await answer_cache.get(cache_key)
    if cached is not None:
        content = cached
    else:
        # Fallback: run once (this will be rare if the stream path ran first)
        logger.debug("No cached answer found for non-streaming request, running LLM once")
        resp = await client.chat_once(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.get("message", {}).get("content", "") or ""
        content = INLINE_SOURCE_RE.sub("", raw)
        # Optionally append sources (same rules as streaming)
        all_sources = collect_sources(chunks)
        sources_block = format_sources_block(list(all_sources.keys()), all_sources)
        if sources_block:
            content = content.rstrip() + "\n\n" + sources_block
        await answer_cache.set(cache_key, content)  # seed cache for future retries

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
