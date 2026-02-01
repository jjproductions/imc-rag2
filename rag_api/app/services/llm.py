import time, json, httpx
from typing import Iterable, Dict, Any, Optional
from app.core.config import settings

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat_stream(self, model: str, messages: list, temperature: float = 0.2, max_tokens: int = 1024) -> Iterable[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True
        }
        with httpx.stream("POST", url, json=payload, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield data
                except Exception:
                    continue

    def chat_once(self, model: str, messages: list, temperature: float = 0.2, max_tokens: int = 1024) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        r = httpx.post(url, json=payload, timeout=None)
        r.raise_for_status()
        return r.json()

def get_ollama() -> OllamaClient:
    return OllamaClient(settings.OLLAMA_BASE_URL)

def aggregate_stream(chunks: Iterable[Dict[str, Any]]) -> str:
    text = []
    for ev in chunks:
        if "message" in ev and "content" in ev["message"]:
            text.append(ev["message"]["content"])
    return "".join(text)