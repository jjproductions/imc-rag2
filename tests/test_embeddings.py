import numpy as np
from rag_api.app.services.embeddings import embed_texts

def test_embedding_shape_and_norm():
    embs = embed_texts(["hello world", "another sentence"])
    assert len(embs.shape) == 2
    assert embs.shape[1] == 1024  # bge-m3
    # normalized
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)