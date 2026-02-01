from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from app.core.config import settings
from app.services.qdrant_client import get_qdrant, ensure_collection
from app.services.embeddings import embed_texts, embed_query

def upsert_payloads(payloads: List[Dict[str, Any]], vectors):
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
        return 0

    # Prepare
    points = []
    for i in to_insert_indices:
        p = payloads[i]
        points.append(
            {
                "id": p["hash"],  # deterministic id for idempotency
                "vector": vectors[i].tolist(),
                "payload": p
            }
        )
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(to_insert_indices)

def search_similar(query: str, top_k: int) -> List[Dict[str, Any]]:
    client = get_qdrant()
    ensure_collection(client)
    q_emb = embed_query(query)
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=q_emb.tolist(),
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
        with_payload=True,
        with_vectors=False
    )
    out = []
    for r in results:
        pl = dict(r.payload)
        pl["score"] = float(r.score)
        out.append(pl)
    return out