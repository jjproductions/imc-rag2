import json, httpx
from typing import Iterable, Dict, Any, AsyncIterator
from app.core.config import settings


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def chat_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        # ignore heartbeats / partials that aren't JSON
                        continue

    async def chat_once(
        self,
        model: str,
        messages: list,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()


def get_ollama() -> OllamaClient:
    return OllamaClient(settings.OLLAMA_BASE_URL)


# not used currently
# def aggregate_stream(chunks: Iterable[Dict[str, Any]]) -> str:
#     text = []
#     for ev in chunks:
#         if "message" in ev and "content" in ev["message"]:
#             text.append(ev["message"]["content"])
#     return "".join(text)
