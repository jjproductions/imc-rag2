import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, BinaryQuantization, BinaryQuantizationConfig, OptimizersConfigDiff, SparseVectorParams
from app.core.config import settings
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    try:
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}:{settings.QDRANT_PORT}")
        client = QdrantClient(
            url=settings.QDRANT_URL, 
            port=settings.QDRANT_PORT, 
            timeout=settings.QDRANT_TIMEOUT
        )
        # Quick check if reachable
        client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise

_collection_ensured = False

def ensure_collection(client: QdrantClient):
    """Check if collection exists and create it if not."""
    global _collection_ensured
    if _collection_ensured:
        return
    try:
        collections = [c.name for c in client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in collections:
            logger.info(f"Collection '{settings.QDRANT_COLLECTION}' not found. Creating...")
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
            logger.info(f"Collection '{settings.QDRANT_COLLECTION}' created successfully.")
        else:
            logger.debug(f"Collection '{settings.QDRANT_COLLECTION}' already exists.")
        _collection_ensured = True
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection: {str(e)}")
        raise