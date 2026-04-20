import json, httpx, logging
from typing import Protocol, Dict, Any, AsyncIterator, List
from app.core.config import settings

from openai import AsyncAzureOpenAI
import time

logger = logging.getLogger(__name__)

class LLMClient(Protocol):
    async def chat_stream(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> AsyncIterator[Dict[str, Any]]:
        ...

    async def chat_once(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        ...

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def chat_stream(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model, "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line: continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        logger.warning(f"Failed to parse Ollama stream line: {line}")
                        continue

    async def chat_once(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model, "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

class AzureOpenAIClient:
    def __init__(self, api_key: str, endpoint: str, api_version: str):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    async def chat_stream(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> AsyncIterator[Dict[str, Any]]:
        # Azure OpenAI stream implementation
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        i = 0
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield {
                    "model": model,
                    "created_at": time.time(),
                    "message": {"role": "assistant", "content": chunk.choices[0].delta.content},
                    "done": False,
                    "index": i
                }
                i += 1
        
        yield { "done": True, "message": {"role": "assistant", "content": ""} }

    async def chat_once(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        return {
            "model": model,
            "created_at": time.time(),
            "message": {"role": "assistant", "content": response.choices[0].message.content},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": response.usage.prompt_tokens if response.usage else 0,
            "eval_count": response.usage.completion_tokens if response.usage else 0,
            "eval_duration": 0
        }



def get_llm_client() -> LLMClient:
    if settings.LLM_PROVIDER == "ollama":
        return OllamaClient(settings.OLLAMA_BASE_URL)
    elif settings.LLM_PROVIDER == "azure_openai":
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")
        return AzureOpenAIClient(
            api_key=settings.AZURE_OPENAI_API_KEY, 
            endpoint=settings.AZURE_OPENAI_ENDPOINT, 
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
