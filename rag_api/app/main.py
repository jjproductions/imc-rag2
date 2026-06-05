import time
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routes import query, stream
from app.services.embeddings import get_models

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading embedding models inside lifespan handler...")
    try:
        # Offload the blocking model loading to a background thread
        await asyncio.to_thread(get_models)
        logger.info("Embedding models pre-loaded successfully.")
    except Exception as e:
        logger.critical(f"Critical error pre-loading embedding models: {e}", exc_info=True)
        raise e
    yield

app = FastAPI(title=settings.APP_NAME, version="1.0.0", lifespan=lifespan)

# Enable CORS so frontends (like OpenWeb UI) can talk to this API.
# Defaults to allowing all origins for quick testing; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def require_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_KEY:
        logger.warning(f"Unauthorized access attempt: provided token '{credentials.credentials[:4]}...' does not match settings.API_KEY")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

# @app.get("/stats", dependencies=[Depends(require_api_key)])
@app.get("/stats")
def stats():
    from app.services.qdrant_client import get_qdrant, ensure_collection
    client = get_qdrant()
    ensure_collection(client)
    info = client.get_collection(settings.QDRANT_COLLECTION)
    count = client.count(collection_name=settings.QDRANT_COLLECTION, exact=True).count
    return JSONResponse({
        "collection": settings.QDRANT_COLLECTION,
        "vectors_count": count,
        "optimizer_status": info.status.value,
        "points_total": info.points_count,
        "indexed_vectors": info.indexed_vectors_count is not None
    })

# Mount routes with auth dependency
# Mount routes with auth dependency
# app.include_router(ingest.router, dependencies=[Depends(require_api_key)])
app.include_router(query.router, dependencies=[Depends(require_api_key)])
app.include_router(stream.router, dependencies=[Depends(require_api_key)])

# OpenAI-compatible Chat Completions mounted at /v1
from app.routes.stream import openai_router, openai_ws_router
app.include_router(openai_router, prefix="/v1", dependencies=[Depends(require_api_key)])
app.include_router(openai_ws_router, prefix="/v1")