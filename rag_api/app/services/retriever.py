from typing import List, Dict, Any
import uuid
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
        # Qdrant requires point IDs to be unsigned ints or UUIDs.
        # Convert deterministic string `hash` into a UUIDv5 for stable, valid IDs.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, p["hash"]))
        points.append(
            {
                "id": point_id,
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
    # Newer qdrant-client uses `query` positional arg for vectors.
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=q_emb.tolist(),
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
        with_payload=True,
        with_vectors=False
    )
    # `query_points` may return a QueryResponse with `.result` or a plain list.
    if hasattr(results, "result"):
        print("Processing results from QueryResponse with .result attribute")
        iterable = results.result
    elif hasattr(results, "points"):
        print("Processing results from QueryResponse with .points attribute")
        iterable = results.points
    else:
        iterable = results
    out = []
    for r in iterable:
        # Handle multiple return shapes from qdrant-client
        # 1) objects with `.payload` and `.score`
        # 2) dicts with 'payload' and 'score'
        # 3) tuples/lists (various orders)
        payload = None
        score = None

        if hasattr(r, "payload"):
            print("Found payload in object with payload attribute")
            payload = dict(r.payload)
            score = getattr(r, "score", None)
        elif isinstance(r, dict):
            payload = dict(r.get("payload", {}))
            score = r.get("score")
        elif isinstance(r, (list, tuple)):
            # Try to locate payload and score inside the tuple
            for item in r:
                if item is None:
                    continue
                if hasattr(item, "payload"):
                    #print to console
                    print("Found payload in item with 1 payload attribute")
                    payload = dict(item.payload)
                elif isinstance(item, dict) and "text" in item:
                    print("Found payload in item dict with text key")
                    payload = dict(item)
                elif isinstance(item, (int, float)):
                    print("Found score in item as int/float")
                    score = float(item)
                elif isinstance(item, dict) and "score" in item:
                    print("Found score in item dict with score key")
                    score = item.get("score")
            if payload is None:
                print("Payload is still None after checking all items in tuple/list")
                # Last-resort: try second element as payload
                try:
                    payload = dict(r[1])
                    print("Found payload in second element of tuple/list")
                except Exception:
                    print("Failed to extract payload from second element of tuple/list")
                    payload = {}

        if payload is None:
            print("Payload is None, skipping this result")
            payload = {}
        try:
            print("Attempting to set score in payload")
            payload["score"] = float(score) if score is not None else float(payload.get("score", 0.0))
        except Exception:
            print("Failed to set score, defaulting to 0.0")
            payload["score"] = 0.0
        out.append(payload)
        print(f"Appended payload with source_id: {payload.get('source_id')} and score: {payload.get('score')}")
    return out