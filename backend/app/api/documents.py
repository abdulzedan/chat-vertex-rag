from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import logging
from datetime import datetime

from app.models.schemas import DocumentUploadResponse, Document, GoogleDriveImportRequest
from app.services.rag_engine import upload_document, list_documents, delete_document, import_from_drive
from app.services.document_uploader import save_uploaded_file, DocumentProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document to the RAG corpus"""
    import time
    request_id = f"{int(time.time())}_{file.filename[:10]}"
    
    try:
        logger.info(f"[{request_id}] Starting upload for file: {file.filename}")
        
        # Validate file type
        allowed_types = ["application/pdf", "image/png", "image/jpeg", "text/csv"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported"
            )
        
        logger.info(f"[{request_id}] File type validated: {file.content_type}")
        
        # Read file content
        content = await file.read()
        logger.info(f"[{request_id}] File content read, size: {len(content)} bytes")
        
        # Save file temporarily
        file_path = await save_uploaded_file(content, file.filename)
        logger.info(f"[{request_id}] File saved temporarily to: {file_path}")
        
        # Upload to RAG corpus
        try:
            logger.info(f"[{request_id}] Attempting to upload to RAG corpus...")
            rag_file = await upload_document(
                file_path=file_path,
                display_name=file.filename,
                description=f"Uploaded {file.content_type}"
            )
            logger.info(f"[{request_id}] Successfully uploaded to RAG corpus: {rag_file.name}")
            document_id = rag_file.name
        except Exception as rag_error:
            logger.error(f"[{request_id}] RAG upload failed: {rag_error}")
            logger.error(f"[{request_id}] Error type: {type(rag_error)}")
            # Fall back to temporary ID
            document_id = f"temp_{file.filename}"
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"[{request_id}] Upload completed successfully for: {file.filename}")
        
        
        return DocumentUploadResponse(
            id=document_id,
            display_name=file.filename,
            file_type=file.content_type,
            upload_time=datetime.now(),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[Document])
async def get_documents():
    """List all documents in the RAG corpus"""
    try:
        # Get actual files from RAG corpus
        files = await list_documents()
        
        documents = []
        for file in files:
            documents.append(Document(
                id=file.name,
                display_name=file.display_name,
                description=file.description,
                create_time=file.create_time,
                update_time=file.update_time,
                size_bytes=file.size_bytes if hasattr(file, 'size_bytes') else None
            ))
        
        logger.info(f"Returning {len(documents)} documents from RAG corpus")
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document from the RAG corpus"""
    try:
        success = await delete_document(document_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import-drive", response_model=DocumentUploadResponse)
async def import_from_google_drive(request: GoogleDriveImportRequest):
    """Import documents from Google Drive URL"""
    import time
    request_id = f"{int(time.time())}_drive_import"
    
    try:
        logger.info(f"[{request_id}] Starting Google Drive import: {request.drive_url}")
        
        # Import from Google Drive
        document_id = await import_from_drive(request.drive_url)
        
        logger.info(f"[{request_id}] Successfully imported from Google Drive: {document_id}")
        
        return DocumentUploadResponse(
            id=document_id,
            display_name="Google Drive Import",
            file_type="google_drive",
            upload_time=datetime.now(),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error importing from Google Drive: {e}")
        raise HTTPException(status_code=500, detail=str(e))