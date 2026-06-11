from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    session_id: str = ''