import threading
import logging
from typing import List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from app.core.config import settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_dense_model = None
_sparse_model = None

def _load_models():
    try:
        logger.info(f"Loading dense embedding model: {settings.EMBEDDING_MODEL}")
        dense = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)
        
        logger.info(f"Loading sparse embedding model: prithivida/Splade_PP_en_v1 (cache_dir={settings.FASTEMBED_CACHE_PATH})")
        sparse = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", cache_dir=settings.FASTEMBED_CACHE_PATH)
        
        return dense, sparse
    except Exception as e:
        logger.error(f"Failed to load embedding models: {str(e)}")
        raise

def get_models():
    global _dense_model, _sparse_model
    if _dense_model is None or _sparse_model is None:
        with _model_lock:
            if _dense_model is None or _sparse_model is None:
                _dense_model, _sparse_model = _load_models()
    return _dense_model, _sparse_model

def embed_texts(texts: List[str]) -> Tuple[np.ndarray, List[Any]]:
    try:
        dense_model, sparse_model = get_models()
        logger.debug(f"Encoding {len(texts)} texts...")
        
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
    except Exception as e:
        logger.error(f"Error during text embedding: {str(e)}")
        raise

def embed_query(text: str) -> Tuple[np.ndarray, Any]:
    try:
        q = text.strip()
        dense, sparse = embed_texts([q])
        return dense[0], sparse[0]
    except Exception as e:
        logger.error(f"Error during query embedding: {str(e)}")
        raise