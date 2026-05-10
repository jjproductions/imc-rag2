from typing import List, Dict, Any, Tuple
import uuid
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams, QuantizationSearchParams, SparseVector, Prefetch, FusionQuery, Fusion
from app.core.config import settings
from app.services.qdrant_client import get_qdrant, ensure_collection
from app.services.embeddings import embed_texts, embed_query

import logging

logger = logging.getLogger(__name__)

def upsert_payloads(payloads: List[Dict[str, Any]], vectors):
    try:
        client = get_qdrant()
        ensure_collection(client)

        # Deduplicate by 'hash'
        to_insert_indices = []
        for i, p in enumerate(payloads):
            flt = Filter(must=[FieldCondition(key="hash", match=MatchValue(value=p["hash"]))])
            scroll = client.scroll(collection_name=settings.QDRANT_COLLECTION, scroll_filter=flt, limit=1)
            if len(scroll[0]) == 0:
                to_insert_indices.append(i)

        if not to_insert_indices:
            logger.info("No new payloads to upsert.")
            return 0

        dense_vectors, sparse_vectors_list = vectors
        # Prepare
        points = []
        for i in to_insert_indices:
            p = payloads[i]
            # Qdrant requires point IDs to be unsigned ints or UUIDs.
            # Convert deterministic string `hash` into a UUIDv5 for stable, valid IDs.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, p["hash"]))
            points.append(
                {
                    "id": point_id,
                    "vector": {
                        "dense": dense_vectors[i].tolist(),
                        "sparse": SparseVector(
                            indices=sparse_vectors_list[i].indices.tolist(),
                            values=sparse_vectors_list[i].values.tolist()
                        )
                    },
                    "payload": p
                }
            )
        
        logger.info(f"Upserting {len(points)} points to '{settings.QDRANT_COLLECTION}'...")
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        logger.info("Upsert completed successfully.")
        return len(to_insert_indices)
    except Exception as e:
        logger.error(f"Failed to upsert payloads: {str(e)}")
        raise

def _extract_payload_and_score(item: Any) -> Tuple[Dict[str, Any], float]:
    """Helper to extract payload and score from various Qdrant return types."""
    payload = {}
    score = 0.0

    # 1. Object with attributes (ScoredPoint)
    if hasattr(item, "payload"):
        payload = dict(item.payload) if item.payload else {}
        score = getattr(item, "score", 0.0) or 0.0
        return payload, score

    # 2. Dictionary
    if isinstance(item, dict):
        payload = dict(item.get("payload", {}))
        score = item.get("score", 0.0)
        return payload, score

    # 3. Tuple/List (legacy or specific query types)
    if isinstance(item, (list, tuple)):
        for sub in item:
            if hasattr(sub, "payload"):
                payload = dict(sub.payload) if sub.payload else {}
            elif isinstance(sub, dict) and "text" in sub:
                payload = dict(sub)
            elif isinstance(sub, (int, float)):
                score = float(sub)
            elif isinstance(sub, dict) and "score" in sub:
                score = sub.get("score", 0.0)
        return payload, score

    return payload, score


def search_similar(query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        client = get_qdrant()
        ensure_collection(client)
        
        logger.debug(f"Searching for similar points: '{query[:50]}...'")
        q_dense, q_sparse = embed_query(query)
        
        # Advanced Hybrid Search with Prefetch and Reciprocal Rank Fusion
        results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(
                    query=q_dense.tolist(),
                    using="dense",
                    limit=top_k * 2,
                    params=SearchParams(
                        hnsw_ef=128,
                        quantization=QuantizationSearchParams(
                            ignore=False,
                            rescore=True,
                            oversampling=3.0
                        )
                    ),
                ),
                Prefetch(
                    query=SparseVector(
                        indices=q_sparse.indices.tolist(),
                        values=q_sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=top_k * 2,
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        # `query_points` may return a QueryResponse with `.result` or a plain list.
        if hasattr(results, "result"):
            iterable = results.result
        elif hasattr(results, "points"):
            iterable = results.points
        else:
            iterable = results

        out = []
        for r in iterable:
            payload, score = _extract_payload_and_score(r)
            
            if not payload:
                logger.warning("Retrieved item with empty payload: %s", r)
                continue
                
            payload["score"] = score
            out.append(payload)
            logger.debug("Retrieved: %s (score=%.4f)", payload.get("source_id"), score)

        return out
    except Exception as e:
        logger.error(f"Error during similar search: {str(e)}")
        raise