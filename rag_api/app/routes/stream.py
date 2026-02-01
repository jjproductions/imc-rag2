        import json, time, uuid, asyncio
        from typing import AsyncGenerator, Dict, Any, Iterable
        from fastapi import APIRouter, Request
        from fastapi.responses import StreamingResponse
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

            async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
                # Stream from Ollama and forward as SSE
                # Each 'data' is JSON: {"delta":"..."}; final contains usage
                async with asyncio.Semaphore(1):
                    for ev in client.chat_stream(
                        model=settings.OLLAMA_MODEL,
                        messages=messages,
                        temperature=settings.TEMPERATURE,
                        max_tokens=settings.MAX_TOKENS
                    ):
                        delta = ev.get("message", {}).get("content", "")
                        if delta:
                            yield {"event": "token", "data": json.dumps({"delta": delta, "trace_id": trace_id})}
                    total_ms = int((time.time() - start) * 1000)
                    usage = {"top_k": top_k, "latency_ms": total_ms}
                    yield {"event": "complete", "data": json.dumps({"complete": True, "usage": usage, "trace_id": trace_id})}

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
            last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            top_k = settings.TOP_K
            chunks = search_similar(last_user, top_k=top_k)
            messages = build_messages(last_user, chunks)
            client = get_ollama()
            model = req.model or settings.OLLAMA_MODEL
            temperature = req.temperature or settings.TEMPERATURE
            max_tokens = req.max_tokens or settings.MAX_TOKENS

            created = int(time.time())
            comp_id = f"chatcmpl-{uuid.uuid4().hex}"

            if req.stream:
                async def gen():
                    # Initial "role" delta per OpenAI spec (optional but common)
                    first_chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(first_chunk)}

"

                    for ev in client.chat_stream(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens):
                        delta = ev.get("message", {}).get("content", "")
                        if delta:
                            data = {
                                "id": comp_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(data)}

"
                    # End of stream
                    done = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(done)}

"
                    yield "data: [DONE]

"

                return StreamingResponse(gen(), media_type="text/event-stream")

            # Non-streaming
            resp = client.chat_once(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            content = resp.get("message", {}).get("content", "")
            data = {
                "id": comp_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            return data