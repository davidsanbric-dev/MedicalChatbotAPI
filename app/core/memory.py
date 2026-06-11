import logging
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List
from app.core.llm import llm
from app.repositories.rps_chat import ChatHistoryRepository

logger = logging.getLogger(__name__)

class PostgreSQLChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.repo = ChatHistoryRepository()

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from PostgreSQL"""
        """Retrieve chat history for a session"""
        try:
            history = self.repo.find_by_session(self.session_id)

            if not history:
                return []

            return messages_from_dict(history.history or [])
        except Exception as e:
            raise Exception(f"Failed to get the chat history: {str(e)}")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Save messages to PostgreSQL.
        Save chat history for a session"""
        # Build the combined history dict from existing messages + new ones
        try:
            existing_messages = self.messages
            all_messages = existing_messages + messages
            history_dict = messages_to_dict(all_messages)

            # Delegate persistence to the repository which manages DB sessions/transactions
            # We pass the serialized history (history_dict). The repository will create or update
            # the ChatHistory row as needed.
            self.repo.save(self.session_id, history_dict)
        except Exception as e:
            # Raise a clearer exception for callers; keep original exception context
            raise Exception(f"Failed to save messages for session {self.session_id}: {e}")

    def clear(self) -> None:
        """Clear messages from PostgreSQL"""
        try:
            self.repo.delete_by_session(self.session_id)
        except Exception as e:
            raise Exception(f"Failed to delete the chat history: {str(e)}")


    async def rephrase_query_with_conversation(self, query: str) -> str:
        """Rewrite the query based on conversation history."""

        query_rephraser_prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="conversation_history"),
            ("user", "Given the previous conversation, generate a search query that is autonomous and can be understood without the history. The query must be about the last topic discussed. Do not answer the question, just rephrase it."),
            ("user", "{input}")
        ])

        conversation_aware_chain = query_rephraser_prompt_template | llm

        response = conversation_aware_chain.invoke({
            "conversation_history": self.messages,
            "input": query
        })

        logger.info("Rephrased query: '%s' -> '%s'", query, response.content)

        return response.content

# Callable to get chat history instance
def chat_message_history(session_id: str) -> PostgreSQLChatMessageHistory:
    return PostgreSQLChatMessageHistory(session_id=session_id)