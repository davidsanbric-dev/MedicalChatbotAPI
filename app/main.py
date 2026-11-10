"""Medical RAG API: station 4 of the medical chatbot pipeline.

Answers health questions from the scraped knowledge base. How it works, visually:
https://medicalchatbot.demos.davieldev.com/api
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.database import prepare_database
from app.core.embedding import warm_up
from app.core.exceptions import public_message
from app.core.rate_limit import limit_chat
from app.routes import chat, insights

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    prepare_database()
    # Without this, the first question would pay the one-time model warm-up.
    logger.info("Warming up the embedding and re-rank models")
    warm_up()
    yield


app = FastAPI(
    title="Medical RAG API",
    description="Answers health questions from scraped condition pages. "
    "[See how it works](https://medicalchatbot.demos.davieldev.com/api).",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def internal_error(request: Request, error: Exception) -> JSONResponse:
    """Any unhandled error: a safe message for the visitor (the server logs the traceback)."""
    return JSONResponse(status_code=500, content={"detail": public_message(error)})


@app.get("/health")
async def health() -> dict[str, str]:
    """For container health checks."""
    return {"status": "healthy"}


# Each chat turn costs two LLM calls, so only chat is rate limited.
app.include_router(chat.router, prefix="/chat", tags=["Chat"], dependencies=[Depends(limit_chat)])
app.include_router(insights.router, prefix="/insights", tags=["Dataset insights"])
