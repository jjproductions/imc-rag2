import pytest
from qdrant_client import QdrantClient
from rag_api.app.core.config import settings
from rag_api.app.services.qdrant_client import get_qdrant, ensure_collection
from rag_api.app.services.embeddings import embed_texts
from rag_api.app.services.retriever import upsert_payloads, search_similar

@pytest.mark.integration
def test_qdrant_roundtrip():
    client = get_qdrant()
    ensure_collection(client)
    payloads = [
        {
            "doc_id": "test_doc",
            "chunk_id": 0,
            "text": "The Institute of Music for Children (IMC) is a nonprofit.",
            "source_path": "/tmp/imc.txt",
            "page": None,
            "hash": "imc-hash-0",
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]
    vectors = embed_texts([payloads[0]["text"]])
    upserted = upsert_payloads(payloads, vectors)
    assert upserted in (0, 1)
    res = search_similar("What is IMC?", top_k=3)
    assert isinstance(res, list)