import threading
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model_lock = threading.Lock()
_model_instance = None

def _load_model():
    model = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)
    return model

def get_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = _load_model()
    return _model_instance

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    embs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalizes
        show_progress_bar=False
    )
    # Ensure 2D float32
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs

def embed_query(text: str) -> np.ndarray:
    # For BGE models, a query instruction can improve results.
    # Keeping generic to avoid network model card pulls.
    q = text.strip()
    return embed_texts([q])[0]