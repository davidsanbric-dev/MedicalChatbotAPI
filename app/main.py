from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.db.database import Base, engine
from app.db import models  # noqa: F401 - ensures models are registered before create_all
from app.routes import chat, insights


@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.begin() as conn:
        # The vector extension must exist before create_all, since the
        # documents table column type depends on it.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy"}

app.include_router(chat.router, prefix="/chat", tags=["Conversational Chat"])
app.include_router(insights.router, prefix="/insights", tags=["Dataset Insights"])
