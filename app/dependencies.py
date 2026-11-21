"""Builds each service once, with the repositories it needs, and hands it to the routes.

Routes receive a service through FastAPI's Depends (e.g. `chat_service: ChatServiceDep`), so a
test swaps it with `app.dependency_overrides` instead of patching modules. The evaluations
(evals/) call these functions directly.
"""

from functools import cache
from typing import Annotated

from fastapi import Depends

from app.repositories.chat_history import ChatHistoryRepository
from app.repositories.documents import DocumentRepository
from app.repositories.handoffs import HandoffRepository
from app.repositories.processed_data import ProcessedDataRepository
from app.services.agent import AgentService
from app.services.chat import ChatService
from app.services.insights import InsightsService
from app.services.retrieval import RetrievalService


@cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(DocumentRepository())


@cache
def get_insights_service() -> InsightsService:
    return InsightsService(ProcessedDataRepository())


@cache
def get_chat_service() -> ChatService:
    return ChatService(get_retrieval_service(), ChatHistoryRepository())


@cache
def get_agent_service() -> AgentService:
    return AgentService(get_retrieval_service(), get_insights_service(), HandoffRepository(), ChatHistoryRepository())


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
InsightsServiceDep = Annotated[InsightsService, Depends(get_insights_service)]
