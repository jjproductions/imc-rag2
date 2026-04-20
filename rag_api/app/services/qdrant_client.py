
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, BinaryQuantization, BinaryQuantizationConfig, OptimizersConfigDiff, SparseVectorParams
from app.core.config import settings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)

def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={"dense": VectorParams(
                size=settings.VECTOR_SIZE,
                distance=Distance.COSINE,
            )},
            sparse_vectors_config={"sparse": SparseVectorParams()},
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=10000,
            ),
        )