import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def mock_lifespan_and_deps():
    """Mock core components to prevent starting real model loading or Qdrant connections."""
    with patch("rag_api.app.main.get_models"), \
         patch("app.services.qdrant_client.get_qdrant") as mock_qdrant:
        yield mock_qdrant

@pytest.fixture(autouse=True)
def clear_caches():
    """Clear global caches before and after each test to ensure test isolation."""
    from app.utils.ttlcache import answer_cache, retrieval_cache
    answer_cache.store.clear()
    retrieval_cache.store.clear()
    yield
    answer_cache.store.clear()
    retrieval_cache.store.clear()

@pytest.fixture
def client():
    # Import the app within mock scope to trigger proper routing initialization
    from rag_api.app.main import app
    return TestClient(app)

MOCK_CHUNKS = [
    {
        "source_id": "leave_policy.pdf",
        "chunk_id": "chunk_1",
        "text": "The annual leave is 15 days.",
        "source_path": "data/docs/leave_policy.pdf",
        "page": 1,
        "score": 0.95,
        "section_path": "Leave"
    }
]

class MockLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def chat_once(self, model, messages, temperature, max_tokens):
        return {
            "message": {"content": self.response_text}
        }

    async def chat_stream(self, model, messages, temperature, max_tokens):
        # Yield chunk by chunk
        words = self.response_text.split(" ")
        for i, word in enumerate(words):
            delta = (" " if i > 0 else "") + word
            yield {
                "message": {"content": delta}
            }

def test_query_normal_response(client):
    with patch("app.routes.query.search_similar") as mock_search, \
         patch("app.routes.query.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        mock_get_llm.return_value = MockLLMClient("The annual leave policy is 15 days.")

        headers = {"Authorization": "Bearer local-key"}
        req_payload = {"question": "What is the policy on leave?"}
        
        response = client.post("/query", json=req_payload, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "15 days" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_id"] == "leave_policy.pdf"

def test_query_dont_know_response(client):
    with patch("app.routes.query.search_similar") as mock_search, \
         patch("app.routes.query.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        headers = {"Authorization": "Bearer local-key"}
        req_payload = {"question": "What is the policy on leave?"}

        # Test "I don't know"
        mock_get_llm.return_value = MockLLMClient("I don't know the answer to that question.")
        response = client.post("/query", json=req_payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "don't know" in data["answer"]
        assert data["sources"] == []

        # Test "I do not know"
        mock_get_llm.return_value = MockLLMClient("I do not know the answer.")
        response = client.post("/query", json=req_payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "do not know" in data["answer"]
        assert data["sources"] == []

def test_chat_completions_non_streaming_normal(client):
    with patch("app.routes.stream.search_similar") as mock_search, \
         patch("app.routes.stream.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        mock_get_llm.return_value = MockLLMClient("The annual leave policy is 15 days.")

        headers = {"Authorization": "Bearer local-key"}
        req_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the policy on leave?"}],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert "15 days" in content
        assert "Sources:" in content
        assert "leave_policy.pdf" in content

def test_chat_completions_non_streaming_dont_know(client):
    with patch("app.routes.stream.search_similar") as mock_search, \
         patch("app.routes.stream.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        mock_get_llm.return_value = MockLLMClient("I don't know what the policy says.")

        headers = {"Authorization": "Bearer local-key"}
        req_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the policy on leave?"}],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert "don't know" in content
        assert "Sources:" not in content
        assert "leave_policy.pdf" not in content

def test_chat_completions_streaming_normal(client):
    with patch("app.routes.stream.search_similar") as mock_search, \
         patch("app.routes.stream.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        mock_get_llm.return_value = MockLLMClient("The annual leave policy is 15 days.")

        headers = {"Authorization": "Bearer local-key"}
        req_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the policy on leave?"}],
            "stream": True
        }
        
        response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
        assert response.status_code == 200
        
        # Parse SSE stream
        lines = response.text.split("\n")
        content_parts = []
        for line in lines:
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    continue
                chunk = json.loads(chunk_str)
                if chunk["choices"][0]["delta"].get("content"):
                    content_parts.append(chunk["choices"][0]["delta"]["content"])
        
        full_content = "".join(content_parts)
        assert "15 days" in full_content
        assert "Sources:" in full_content
        assert "leave_policy.pdf" in full_content

def test_chat_completions_streaming_dont_know(client):
    with patch("app.routes.stream.search_similar") as mock_search, \
         patch("app.routes.stream.get_llm_client") as mock_get_llm:
        mock_search.return_value = MOCK_CHUNKS
        mock_get_llm.return_value = MockLLMClient("I do not know the answer.")

        headers = {"Authorization": "Bearer local-key"}
        req_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the policy on leave?"}],
            "stream": True
        }
        
        response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
        assert response.status_code == 200
        
        # Parse SSE stream
        lines = response.text.split("\n")
        content_parts = []
        for line in lines:
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    continue
                chunk = json.loads(chunk_str)
                if chunk["choices"][0]["delta"].get("content"):
                    content_parts.append(chunk["choices"][0]["delta"]["content"])
        
        full_content = "".join(content_parts)
        assert "do not know" in full_content
        assert "Sources:" not in full_content
        assert "leave_policy.pdf" not in full_content
