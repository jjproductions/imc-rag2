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

@pytest.fixture
def client():
    # Import the app within mock scope to trigger proper routing initialization
    from rag_api.app.main import app
    return TestClient(app)

@patch("rag_api.app.routes.query.search_similar")
def test_query_endpoint_fallback(mock_search, client):
    # Mock search_similar to raise an exception simulating database failure
    mock_search.side_effect = Exception("Qdrant collection not found")

    headers = {"Authorization": "Bearer local-key"}
    req_payload = {"question": "What is the policy on leave?", "top_k": 3}
    
    response = client.post("/query", json=req_payload, headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "I am sorry, but the document database is temporarily unavailable" in data["answer"]
    assert data["sources"] == []
    assert data["usage"]["top_k"] == 3

@patch("rag_api.app.routes.stream.search_similar")
def test_stream_endpoint_fallback(mock_search, client):
    mock_search.side_effect = Exception("Qdrant collection not found")

    headers = {"Authorization": "Bearer local-key"}
    req_payload = {"question": "What is the policy on leave?", "top_k": 3}
    
    response = client.post("/stream", json=req_payload, headers=headers)
    assert response.status_code == 200
    
    # Parse SSE events
    lines = response.text.split("\n")
    events = []
    current_event = {}
    for line in lines:
        if line.startswith("event:"):
            current_event["event"] = line.split("event:")[1].strip()
        elif line.startswith("data:"):
            current_event["data"] = line.split("data:")[1].strip()
            events.append(current_event)
            current_event = {}

    assert len(events) >= 2
    assert events[0]["event"] == "token"
    token_data = json.loads(events[0]["data"])
    assert "I am sorry, but the document database is temporarily unavailable" in token_data["delta"]

    assert events[1]["event"] == "complete"
    complete_data = json.loads(events[1]["data"])
    assert complete_data["complete"] is True

@patch("rag_api.app.routes.stream.search_similar")
def test_chat_completions_non_streaming_fallback(mock_search, client):
    mock_search.side_effect = Exception("Qdrant collection not found")

    headers = {"Authorization": "Bearer local-key"}
    req_payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the policy on leave?"}],
        "stream": False
    }
    
    response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "I am sorry, but the document database is temporarily unavailable. Please try again later."
    assert data["choices"][0]["finish_reason"] == "stop"

@patch("rag_api.app.routes.stream.search_similar")
def test_chat_completions_streaming_fallback(mock_search, client):
    mock_search.side_effect = Exception("Qdrant collection not found")

    headers = {"Authorization": "Bearer local-key"}
    req_payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the policy on leave?"}],
        "stream": True
    }
    
    response = client.post("/v1/chat/completions", json=req_payload, headers=headers)
    assert response.status_code == 200
    
    # Parse SSE stream chunk by chunk
    lines = response.text.split("\n")
    chunks = []
    for line in lines:
        if line.startswith("data: "):
            chunk_str = line[6:].strip()
            if chunk_str == "[DONE]":
                continue
            chunks.append(json.loads(chunk_str))

    assert len(chunks) >= 3
    # Chunk 1: Role
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # Chunk 2: Error content
    assert "I am sorry, but the document database is temporarily unavailable" in chunks[1]["choices"][0]["delta"]["content"]
    # Chunk 3: Stop
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
