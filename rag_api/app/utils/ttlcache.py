import time
import asyncio, json
import numpy as np
from collections import OrderedDict
from typing import Any, Optional, Tuple


class SemanticTTLCache:
    def __init__(self, maxsize: int = 512, ttl_seconds: int = 300, similarity_threshold: float = 0.95):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.similarity_threshold = similarity_threshold
        # store: { hash_key: (timestamp, vector_emb, messages_context, value) }
        self.store: "OrderedDict[str, Tuple[float, Any, str, Any]]" = OrderedDict()
        self.lock = asyncio.Lock()

    def _purge_expired(self):
        now = time.time()
        keys_to_delete = [k for k, (ts, _, _, _) in self.store.items() if now - ts > self.ttl]
        for k in keys_to_delete:
            self.store.pop(k, None)

    async def get_semantically(self, query_emb: Any, context_str: str) -> Optional[Any]:
        # Fast brute-force cosine similarity over at most 512 items
        async with self.lock:
            self._purge_expired()
            
            best_match_key = None
            best_score = -1
            
            for k, (ts, v_emb, v_context, val) in self.store.items():
                if v_context != context_str:
                    continue # only match if system/history parameters are identical
                
                # Cosine similarity (assuming normalized vectors)
                score = float(np.dot(query_emb, v_emb))
                if score > best_score:
                    best_score = score
                    best_match_key = k
            
            if best_match_key and best_score >= self.similarity_threshold:
                # Cache Hit! Refresh LRU
                ts, v_emb, v_context, val = self.store.pop(best_match_key)
                self.store[best_match_key] = (ts, v_emb, v_context, val)
                print(f"Semantic Cache Hit! Score: {best_score:.4f}")
                return val
                
            return None

    def _normalize_key(self, key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, dict):
            return json.dumps(key, sort_keys=True)
        return str(key)

    async def get(self, key: Any) -> Optional[Any]:
        normalized_key = self._normalize_key(key)
        async with self.lock:
            self._purge_expired()
            if normalized_key in self.store:
                ts, v_emb, c_str, val = self.store.pop(normalized_key)
                self.store[normalized_key] = (ts, v_emb, c_str, val)
                return val
            return None

    async def set(self, key: Any, value: Any, emb: Any = None, context_str: str = ""):
        normalized_key = self._normalize_key(key)
        async with self.lock:
            self._purge_expired()
            if normalized_key in self.store:
                self.store.pop(normalized_key, None)
            
            self.store[normalized_key] = (time.time(), emb, context_str, value)
            
            while len(self.store) > self.maxsize:
                self.store.popitem(last=False)

    async def invalidate_prefix(self, prefix: str):
        # Optional helper if you want to drop a subset
        async with self.lock:
            for k in list(self.store.keys()):
                if k.startswith(prefix):
                    self.store.pop(k, None)


# Global caches
answer_cache = SemanticTTLCache(
    maxsize=512, ttl_seconds=600, similarity_threshold=0.96
)  # final assistant message per request

retrieval_cache = SemanticTTLCache(maxsize=512, ttl_seconds=600)  # optional: cached RAG chunks
