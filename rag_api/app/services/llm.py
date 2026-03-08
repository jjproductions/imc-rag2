import json, httpx, logging
from typing import Protocol, Dict, Any, AsyncIterator, List
from app.core.config import settings

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

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

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    async def chat_stream(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> AsyncIterator[Dict[str, Any]]:
        gemini_model = genai.GenerativeModel(model)
        generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        
        # Gemini expects a list of alternating user/model roles.
        # We assume the last message is the user prompt.
        prompt = messages[-1]['content']
        
        stream = await gemini_model.generate_content_async(
            prompt,
            stream=True,
            generation_config=generation_config
        )
        
        # Adapt Gemini stream to Ollama-like dictionary structure
        i = 0
        async for chunk in stream:
            if chunk.parts:
                yield {
                    "model": model,
                    "created_at": chunk.created_at.isoformat(),
                    "message": {"role": "assistant", "content": chunk.text},
                    "done": False,
                    "index": i
                }
                i += 1
        
        # Yield a final "done" message
        yield { "done": True, "message": {"role": "assistant", "content": ""} }


    async def chat_once(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        gemini_model = genai.GenerativeModel(model)
        generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        
        prompt = messages[-1]['content']
        
        response = await gemini_model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        # Adapt Gemini response to Ollama-like dictionary structure
        return {
            "model": model,
            "created_at": response.created_at.isoformat(),
            "message": {"role": "assistant", "content": response.text},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0
        }


def get_llm_client() -> LLMClient:
    if settings.LLM_PROVIDER == "ollama":
        return OllamaClient(settings.OLLAMA_BASE_URL)
    elif settings.LLM_PROVIDER == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set to use the Gemini provider.")
        return GeminiClient(settings.GEMINI_API_KEY)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
