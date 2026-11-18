"""Medical RAG API: station 4 of the medical chatbot pipeline.

Answers health questions from the scraped knowledge base. How it works, visually:
https://medicalchatbot.demos.davieldev.com/api
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import LOG_LEVEL
from app.core.database import prepare_database
from app.core.embedding import warm_up
from app.core.exceptions import public_message
from app.routes import agent, chat, insights
from app.routes.rate_limit import limit_chat
from app.schemas import ErrorResponse

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def json_error(description: str) -> dict:
    """An error documented as an ErrorResponse JSON body. Stated as explicit content, not
    `"model"`, because FastAPI files a model under the route's media type, which for the
    streams is text/event-stream, while their errors are always JSON."""
    return {"description": description, "content": {"application/json": {"schema": ErrorResponse.model_json_schema()}}}


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
    responses={500: json_error("Something failed; the message is safe to show")},
)


@app.exception_handler(Exception)
async def internal_error(request: Request, error: Exception) -> JSONResponse:
    """Any unhandled error: a safe message for the visitor (the server logs the traceback)."""
    return JSONResponse(status_code=500, content=ErrorResponse(detail=public_message(error)).model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    """For container health checks."""
    return {"status": "healthy"}


# Each chat turn costs two LLM calls and an agent turn up to four, so both share the chat limits.
rate_limited = {
    "dependencies": [Depends(limit_chat)],
    "responses": {429: json_error("A rate limit was reached; see Retry-After")},
}
app.include_router(chat.router, prefix="/chat", tags=["Chat"], **rate_limited)
app.include_router(agent.router, prefix="/agent", tags=["Agent"], **rate_limited)
app.include_router(insights.router, prefix="/insights", tags=["Dataset insights"])
