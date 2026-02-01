import os
from typing import List
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.models.schemas import IngestRequest
from app.utils.chunking import recursive_find_files, read_text_from_file, build_payloads
from app.services.embeddings import embed_texts
from app.services.retriever import upsert_payloads

router = APIRouter(prefix="", tags=["ingestion"])

@router.post("/ingest")
def ingest(req: IngestRequest):
    # Accept absolute host paths; inside container, map your host path under /data via volume
    path = req.path
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")

    files = []
    if os.path.isdir(path):
        files = recursive_find_files(path)
    else:
        files = [path]

    all_payloads = []
    all_texts = []
    for f in files:
        try:
            text, _ = read_text_from_file(f)
            pls = build_payloads(f, text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            all_payloads.extend(pls)
            all_texts.extend([p["text"] for p in pls])
        except ValueError as e:
            # skip unsupported
            continue

    if not all_payloads:
        return {"inserted": 0, "skipped": 0}

    vectors = embed_texts(all_texts)
    inserted = upsert_payloads(all_payloads, vectors)
    skipped = len(all_payloads) - inserted
    return {"inserted": inserted, "skipped": skipped}