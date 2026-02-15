import time
import asyncio, json
from collections import OrderedDict
from typing import Any, Optional, Tuple


class TTLCache:
    def __init__(self, maxsize: int = 512, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self.lock = asyncio.Lock()

    def _purge_expired(self):
        now = time.time()
        keys_to_delete = []
        for k, (ts, _) in list(self.store.items()):
            if now - ts > self.ttl:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            self.store.pop(k, None)

    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            self._purge_expired()
            print(
                f"Cache get: {key}. Current keys: {json.dumps(list(self.store.keys()), indent=2)}"
            )
            if key in self.store:
                ts, val = self.store.pop(key)
                # move to end (LRU)
                self.store[key] = (ts, val)
                return val
            return None

    async def set(self, key: str, value: Any):
        async with self.lock:
            self._purge_expired()
            if key in self.store:
                self.store.pop(key, None)
            self.store[key] = (time.time(), value)
            print(
                f"Cache set: {key}. Current keys: {json.dumps(list(self.store.keys()), indent=2)}"
            )
            # Enforce LRU maxsize
            while len(self.store) > self.maxsize:
                self.store.popitem(last=False)

    async def invalidate_prefix(self, prefix: str):
        # Optional helper if you want to drop a subset
        async with self.lock:
            for k in list(self.store.keys()):
                if k.startswith(prefix):
                    self.store.pop(k, None)


# Global caches
answer_cache = TTLCache(
    maxsize=512, ttl_seconds=600
)  # final assistant message per request
retrieval_cache = TTLCache(maxsize=512, ttl_seconds=600)  # optional: cached RAG chunks
