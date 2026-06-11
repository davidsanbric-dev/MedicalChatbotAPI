from app.core.database import session_scope
from app.core.models import ChatHistory

class ChatHistoryRepository:
    def find_by_session(self, session_id: str):
        """Retrieve chat history for a session"""
        with session_scope() as db:
            return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).first()

    def save(self, session_id: str, history):
        """Create or update the history row for a session."""
        with session_scope() as db:
            chat_history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).first()
            if chat_history:
                chat_history.history = history
            else:
                chat_history = ChatHistory(session_id=session_id, history=history)
                db.add(chat_history)
            return chat_history

    def delete_by_session(self, session_id):
        """Clear chat history for a session"""
        with session_scope() as db:
            db.query(ChatHistory).filter(
                ChatHistory.session_id == session_id
            ).delete()