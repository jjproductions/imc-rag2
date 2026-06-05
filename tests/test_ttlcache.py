import pytest
import asyncio

try:
    from app.utils.ttlcache import SemanticTTLCache
except ModuleNotFoundError:
    from rag_api.app.utils.ttlcache import SemanticTTLCache


def test_ttlcache_dict_key():
    cache = SemanticTTLCache(maxsize=10, ttl_seconds=60)
    
    # Define a dictionary key
    key1 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    # Define an identical dictionary key but with different key insertion order
    key2 = {
        "max_tokens": 100,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4"
    }
    
    async def run_test():
        # Set value
        await cache.set(key1, "cached_value_1")
        
        # Retrieve using exact same key
        val1 = await cache.get(key1)
        assert val1 == "cached_value_1"
        
        # Retrieve using key2 (which is semantically identical but has different key order)
        val2 = await cache.get(key2)
        assert val2 == "cached_value_1"

    asyncio.run(run_test())


def test_ttlcache_string_key():
    cache = SemanticTTLCache(maxsize=10, ttl_seconds=60)
    
    async def run_test():
        await cache.set("test_key", "test_value")
        val = await cache.get("test_key")
        assert val == "test_value"

    asyncio.run(run_test())
