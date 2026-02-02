from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class IngestRequest(BaseModel):
    path: str  # server-visible path (e.g., /data/docs)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class StreamRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    trace_id: Optional[str] = None

class RetrievedChunk(BaseModel):
    source_id: str
    chunk_id: str
    text: str
    source_path: str
    page: Optional[int] = None
    score: Optional[float] = None
    section: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    usage: Dict[str, Any] = {}

# --- OpenAI-compatible schemas ---
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None