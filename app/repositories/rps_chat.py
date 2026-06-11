from app.core.database import SessionLocal
from app.core.models import ChatHistory

class ChatHistoryRepository:
    def __init__(self):
        pass

    def find_by_session(self, session_id: str):
        """Retrieve chat history for a session"""
        db = SessionLocal()
        try:
            return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).first()
        finally:
            db.close()

    def save(self, session_id: str, history):
        if SessionLocal is None or ChatHistory is None:
            raise RuntimeError("Database not configured: SessionLocal or ChatHistory is missing")

        # Basic validation for history can be added here if your model expects a specific type
        # e.g. ensure it's JSON-serializable or a list/dict

        # Use context-managed session to ensure proper close(), and handle rollback on exceptions
        with SessionLocal() as db:
            try:
                chat_history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).first()
                if chat_history:
                    chat_history.history = history
                else:
                    chat_history = ChatHistory(session_id=session_id, history=history)
                    db.add(chat_history)

                db.commit()

                # Ensure the instance has fresh attributes after commit
                try:
                    db.refresh(chat_history)
                except Exception:
                    # refresh may fail for some detached objects; ignore but keep commit
                    pass

                return chat_history
            except Exception:
                db.rollback()
                raise

    def delete_by_session(self, session_id):
        """Clear chat history for a session"""
        db = SessionLocal()
        try:
            db.query(ChatHistory).filter(
                ChatHistory.session_id == session_id
            ).delete()
            db.commit()
        finally:
            db.close()