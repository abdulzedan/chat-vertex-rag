from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    id: str
    display_name: str
    file_type: str
    upload_time: datetime
    status: str = "success"

class Document(BaseModel):
    id: str
    display_name: str
    description: Optional[str] = None
    create_time: datetime
    update_time: datetime
    size_bytes: Optional[int] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    use_rag: bool = True
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    
class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    similarity_top_k: int = 10
    vector_distance_threshold: float = 0.5
    session_id: Optional[str] = None

class GoogleDriveImportRequest(BaseModel):
    drive_url: str