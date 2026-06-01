import numpy as np
from rag_api.app.services.embeddings import embed_texts

def test_embedding_shape_and_norm():
    dense, sparse = embed_texts(["hello world", "another sentence"])
    assert len(dense.shape) == 2
    assert dense.shape[1] == 1024  # bge-m3
    # normalized
    norms = np.linalg.norm(dense, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    
    # Verify sparse
    assert len(sparse) == 2
    assert hasattr(sparse[0], "indices")
    assert hasattr(sparse[0], "values")