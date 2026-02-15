
import pytest
from unittest.mock import MagicMock, patch
from rag_api.app.services.retriever import search_similar

# Mock the dependencies
@pytest.fixture
def mock_qdrant():
    with patch("rag_api.app.services.retriever.get_qdrant") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_settings():
    with patch("rag_api.app.services.retriever.settings") as mock_settings:
        mock_settings.QDRANT_COLLECTION = "test_collection"
        yield mock_settings

@pytest.fixture
def mock_embed():
    with patch("rag_api.app.services.retriever.embed_query") as mock_embed:
        mock_embed.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        yield mock_embed

def test_search_similar_handles_payload_objects(mock_qdrant, mock_settings, mock_embed):
    # Case 1: List of objects with payload attribute (ScoredPoint-like)
    mock_point = MagicMock()
    mock_point.payload = {"source_id": "doc1", "text": "content"}
    mock_point.score = 0.9
    
    mock_qdrant.query_points.return_value = [mock_point]
    
    results = search_similar("query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["source_id"] == "doc1"
    assert results[0]["score"] == 0.9

def test_search_similar_handles_dicts(mock_qdrant, mock_settings, mock_embed):
    # Case 2: List of dicts
    mock_qdrant.query_points.return_value = [
        {"payload": {"source_id": "doc2", "text": "content"}, "score": 0.8}
    ]
    
    results = search_similar("query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["source_id"] == "doc2"
    assert results[0]["score"] == 0.8

def test_search_similar_handles_tuples(mock_qdrant, mock_settings, mock_embed):
    # Case 3: List of tuples (legacy)
    # e.g. (payload_object, score) or (score, payload_dict)
    
    # Let's verify our logic for tuple: it iterates and finds payload/score
    mock_payload = {"source_id": "doc3", "text": "content"}
    mock_qdrant.query_points.return_value = [
        (0.75, mock_payload) 
    ]
    
    results = search_similar("query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["source_id"] == "doc3"
    assert results[0]["score"] == 0.75

def test_search_similar_handles_query_response_object(mock_qdrant, mock_settings, mock_embed):
    # Handles object with .result or .points
    class MockResponse:
        def __init__(self):
            self.points = []
            
    mock_response = MockResponse()
    mock_point = MagicMock()
    mock_point.payload = {"source_id": "doc4", "text": "content"}
    mock_point.score = 0.95
    mock_response.points = [mock_point]
    
    mock_qdrant.query_points.return_value = mock_response
    
    results = search_similar("query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["source_id"] == "doc4"
