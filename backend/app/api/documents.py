from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import logging
import uuid
from datetime import datetime

from app.models.schemas import DocumentUploadResponse, Document, GoogleDriveImportRequest
from app.services.document_processor import DocumentProcessor
from app.services.enhanced_document_processor import EnhancedDocumentProcessor
from app.services.vertex_search import VertexSearchService
from app.services.document_uploader import save_uploaded_file

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
logger.info("Initializing document services...")
document_processor = DocumentProcessor()
enhanced_processor = EnhancedDocumentProcessor()
search_service = VertexSearchService()
logger.info("Document services initialized")

# Always use enhanced processor for better document processing
USE_ENHANCED_PROCESSOR = True

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document instantly"""
    import time
    request_id = f"{int(time.time())}_{file.filename[:10]}"
    
    try:
        logger.info(f"[{request_id}] Starting fast upload for file: {file.filename}")
        
        # Validate file type - expanded support
        allowed_types = [
            "application/pdf", 
            "image/png", "image/jpeg", "image/jpg",
            "text/csv", "application/csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
            "text/plain"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported. Supported: PDF, Images, CSV, Word, Excel, Text"
            )
        
        logger.info(f"[{request_id}] File type validated: {file.content_type}")
        
        # Check if file already exists before processing
        existing_docs = await search_service.list_documents()
        for doc in existing_docs:
            if doc['filename'] == file.filename:
                logger.info(f"[{request_id}] File already exists, skipping: {file.filename}")
                # Return success response but indicate file was skipped
                return DocumentUploadResponse(
                    id=doc['id'],  # Use existing document ID
                    display_name=file.filename,
                    file_type=file.content_type,
                    upload_time=datetime.now(),
                    status="skipped - already exists"
                )
        
        # Read file content
        content = await file.read()
        logger.info(f"[{request_id}] File content read, size: {len(content)} bytes")
        
        # Save file temporarily
        file_path = await save_uploaded_file(content, file.filename)
        logger.info(f"[{request_id}] File saved temporarily to: {file_path}")
        
        # Process document to extract text
        try:
            logger.info(f"[{request_id}] Processing document...")
            
            # Use enhanced processor if enabled
            if USE_ENHANCED_PROCESSOR:
                processed_doc = await enhanced_processor.process_file(
                    file_path=file_path,
                    file_type=file.content_type,
                    filename=file.filename
                )
                logger.info(f"[{request_id}] Enhanced processing complete: {processed_doc['chunk_count']} semantic chunks")
                logger.info(f"[{request_id}] Metadata extracted: {len(processed_doc['metadata']['sections'])} sections, {len(processed_doc['tables'])} tables")
            else:
                processed_doc = await document_processor.process_file(
                    file_path=file_path,
                    file_type=file.content_type,
                    filename=file.filename
                )
            
            logger.info(f"[{request_id}] Document processed: {processed_doc['chunk_count']} chunks, {processed_doc['character_count']} characters")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Prepare metadata with enhanced information
            metadata = {
                'file_type': file.content_type,
                'upload_time': datetime.now().isoformat(),
                'chunk_count': processed_doc['chunk_count'],
                'char_count': processed_doc['character_count']
            }
            
            # Add enhanced metadata if available
            if USE_ENHANCED_PROCESSOR and 'metadata' in processed_doc:
                metadata.update({
                    'word_count': processed_doc['metadata']['word_count'],
                    'has_tables': processed_doc['metadata']['has_tables'],
                    'sections': processed_doc['metadata']['sections'][:5],  # Top 5 sections
                    'entities': processed_doc['metadata']['entities']
                })
            
            # Index in Vertex AI Search
            logger.info(f"[{request_id}] Indexing in Vertex AI Search...")
            await search_service.index_document(
                document_id=document_id,
                filename=file.filename,
                text_content=processed_doc['full_text'],
                chunks=processed_doc['chunks'],
                metadata=metadata
            )
            logger.info(f"[{request_id}] Document indexed successfully!")
            
        except Exception as process_error:
            logger.error(f"[{request_id}] Document processing failed: {process_error}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(process_error)}")
        
        finally:
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[Document])
async def get_documents():
    """List all documents in Vertex AI Search"""
    try:
        # Get documents from Vertex AI Search
        search_docs = await search_service.list_documents()
        
        documents = []
        for doc in search_docs:
            documents.append(Document(
                id=doc['id'],
                display_name=doc['filename'],
                description=f"{doc['file_type']} - {doc.get('chunk_count', 0)} chunks",
                create_time=datetime.fromisoformat(doc['upload_time']) if doc.get('upload_time') else datetime.now(),
                update_time=datetime.fromisoformat(doc['upload_time']) if doc.get('upload_time') else datetime.now(),
                size_bytes=doc.get('character_count', 0)
            ))
        
        logger.info(f"Returning {len(documents)} documents from Vertex AI Search")
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document from Vertex AI Search"""
    try:
        success = await search_service.delete_document(document_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def clear_all_documents():
    """Clear all documents from Vertex AI Search"""
    try:
        logger.info("Received request to clear all documents")
        success = await search_service.clear_all_documents()
        
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear some documents")
            
    except Exception as e:
        logger.error(f"Error clearing all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import-drive", response_model=DocumentUploadResponse)
async def import_from_google_drive(request: GoogleDriveImportRequest):
    """Import documents from Google Drive URL using fast processing"""
    import time
    request_id = f"{int(time.time())}_drive_import"
    
    try:
        logger.info(f"[{request_id}] Starting fast Google Drive import: {request.drive_url}")
        
        # For now, return an informative message about the new approach
        # This will need to be implemented to download from Google Drive first,
        # then process using our fast document processor
        raise HTTPException(
            status_code=501, 
            detail="Google Drive import is being updated for the new fast processing system. Please download the file and upload it directly for now."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error importing from Google Drive: {e}")
        raise HTTPException(status_code=500, detail=str(e))