import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.database import Base, engine
from app.core import models  # noqa: F401 - ensures models are registered before create_all
from app.core.embedding import encode_to_embedding, generate_relevance_scores
from app.core.exceptions import ChatHistoryError, RetrievalError
from app.routes import chat, insights

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def ensure_embedding_index():
    """Bring documents_embedding_idx in line with the model definition.

    create_all only builds indexes for tables it creates, so databases made
    before the switch to HNSW still carry the old ivfflat index (built on an
    empty table, with the default L2 opclass that the cosine search can't
    use). Replace it once; afterwards this is a no-op.
    """
    index = next(i for i in models.Document.__table__.indexes if i.name == "documents_embedding_idx")
    with engine.begin() as conn:
        indexdef = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": index.name}
        ).scalar()
        if indexdef and ("USING hnsw" not in indexdef or "vector_cosine_ops" not in indexdef):
            logger.info("Replacing outdated vector index: %s", indexdef)
            conn.execute(text(f"DROP INDEX {index.name}"))
        index.create(bind=conn, checkfirst=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.begin() as conn:
        # The vector extension must exist before create_all, since the
        # documents table column type depends on it.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(bind=engine)
    ensure_embedding_index()

    # Warm up the ONNX embedding and cross-encoder sessions so the first real
    # /chat/ request doesn't pay the one-time inference warmup (which otherwise
    # pushes the first request past the frontend's request timeout).
    logger.info("Warming up embedding and reranker models...")
    encode_to_embedding("warmup")
    generate_relevance_scores([("warmup query", "warmup passage")])
    logger.info("Model warmup complete")

    yield

app = FastAPI(lifespan=lifespan)


# Map service-layer failures to HTTP responses once, so route handlers can stay
# focused on the happy path. (The /chat/stream SSE endpoint handles errors inline
# instead, since you can't raise HTTPException mid-stream.)
@app.exception_handler(RetrievalError)
@app.exception_handler(ChatHistoryError)
@app.exception_handler(RuntimeError)
async def _internal_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled %s on %s", type(exc).__name__, request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy"}

app.include_router(chat.router, prefix="/chat", tags=["Conversational Chat"])
app.include_router(insights.router, prefix="/insights", tags=["Dataset Insights"])
