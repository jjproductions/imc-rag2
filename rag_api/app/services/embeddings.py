import threading
from typing import List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from app.core.config import settings

_model_lock = threading.Lock()
_dense_model = None
_sparse_model = None

def _load_models():
    dense = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)
    sparse = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return dense, sparse

def get_models():
    global _dense_model, _sparse_model
    if _dense_model is None or _sparse_model is None:
        with _model_lock:
            if _dense_model is None or _sparse_model is None:
                _dense_model, _sparse_model = _load_models()
    return _dense_model, _sparse_model

def embed_texts(texts: List[str]) -> Tuple[np.ndarray, List[Any]]:
    dense_model, sparse_model = get_models()
    embs = dense_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalizes
        show_progress_bar=False
    )
    
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
        
    sparse_list = list(sparse_model.embed(texts, batch_size=32))

    return embs, sparse_list

def embed_query(text: str) -> Tuple[np.ndarray, Any]:
    q = text.strip()
    dense, sparse = embed_texts([q])
    return dense[0], sparse[0]