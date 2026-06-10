from typing import Dict
# from typing import Optional
from pydantic import BaseModel


class DocumentRequest(BaseModel):
    file_path: str
    # content: str
    # metadata_: Dict = {}


class RetrievalRequest(BaseModel):
    query: str
    session_id: str = ''
    # top_k: int = 10
    # image_base64: Optional[str] = None