import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.core.database import Base, engine
from app.core import models  # noqa: F401 - ensures models are registered before create_all
from app.core.embedding import encode_to_embedding, generate_relevance_scores
from app.routes import chat, insights

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.begin() as conn:
        # The vector extension must exist before create_all, since the
        # documents table column type depends on it.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(bind=engine)

    # Warm up the ONNX embedding and cross-encoder sessions so the first real
    # /chat/ request doesn't pay the one-time inference warmup (which otherwise
    # pushes the first request past the frontend's request timeout).
    logger.info("Warming up embedding and reranker models...")
    encode_to_embedding("warmup")
    generate_relevance_scores([("warmup query", "warmup passage")])
    logger.info("Model warmup complete")

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy"}

app.include_router(chat.router, prefix="/chat", tags=["Conversational Chat"])
app.include_router(insights.router, prefix="/insights", tags=["Dataset Insights"])
