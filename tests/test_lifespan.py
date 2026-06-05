import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_lifespan_preloads_models():
    # Mock `get_models` to avoid loading real models during the startup test
    with patch("rag_api.app.main.get_models") as mock_get_models:
        # Import the app inside the patch scope so its creation/lifespan is tested
        from rag_api.app.main import app
        
        # TestClient using context manager triggers the lifespan startup and shutdown
        with TestClient(app) as client:
            # Check that a basic request works (health check)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.text == "ok"
            
        # Verify that get_models was called exactly once during startup
        mock_get_models.assert_called_once()
