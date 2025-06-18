from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import logging
import uuid
from datetime import datetime

from app.models.schemas import DocumentUploadResponse, Document, GoogleDriveImportRequest
from app.services.enhanced_document_processor import EnhancedDocumentProcessor
from app.services.vertex_search import VertexSearchService
from app.services.document_uploader import save_uploaded_file

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
logger.info("Initializing document services...")
enhanced_processor = EnhancedDocumentProcessor()
search_service = VertexSearchService()
logger.info("Document services initialized")

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
            
            # Process document using enhanced processor
            processed_doc = await enhanced_processor.process_file(
                file_path=file_path,
                file_type=file.content_type,
                filename=file.filename
            )
            logger.info(f"[{request_id}] Enhanced processing complete: {processed_doc['chunk_count']} semantic chunks")
            logger.info(f"[{request_id}] Metadata extracted: {len(processed_doc['metadata']['sections'])} sections, {len(processed_doc['tables'])} tables")
            
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
            
            # Add enhanced metadata
            if 'metadata' in processed_doc:
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
        logger.info(f"Received delete request for document ID: {document_id}")
        success = await search_service.delete_document(document_id)
        if success:
            logger.info(f"Successfully deleted document: {document_id}")
            return {"message": "Document deleted successfully"}
        else:
            logger.warning(f"Document not found: {document_id}")
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
    import re
    import httpx
    import tempfile
    import os
    
    request_id = f"{int(time.time())}_drive_import"
    
    try:
        logger.info(f"[{request_id}] Starting Google Drive import: {request.drive_url}")
        
        # Extract file ID and type from Google Drive URL
        file_id, doc_type = _extract_drive_file_id(request.drive_url)
        if not file_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid Google Drive URL. Please use a shareable Google Drive link."
            )
        
        logger.info(f"[{request_id}] Extracted file ID: {file_id}, type: {doc_type}")
        
        # Handle different Google Drive document types
        if doc_type in ['document', 'spreadsheet', 'presentation']:
            # For Google Docs/Sheets/Slides, use export endpoints
            if doc_type == 'document':
                download_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
                expected_extension = '.pdf'
                expected_content_type = 'application/pdf'
            elif doc_type == 'spreadsheet':
                download_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
                expected_extension = '.xlsx'
                expected_content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            else:  # presentation
                download_url = f"https://docs.google.com/presentation/d/{file_id}/export?format=pdf"
                expected_extension = '.pdf'
                expected_content_type = 'application/pdf'
            
            logger.info(f"[{request_id}] Using export URL for {doc_type}: {download_url}")
        else:
            # For regular files, use download endpoint
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            expected_extension = None
            expected_content_type = None
        
        # Try multiple download methods for Google Drive
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            logger.info(f"[{request_id}] Downloading file from Google Drive...")
            
            # Try the appropriate download method
            response = await client.get(download_url)
            
            # Check if we got redirected to login (indicates private file)
            if "accounts.google.com" in str(response.url) or response.status_code != 200:
                logger.info(f"[{request_id}] Download failed, trying alternative methods...")
                
                if doc_type == 'file':
                    # For regular files, try with confirmation token for large files
                    alt_download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    response = await client.get(alt_download_url)
                    
                    if "accounts.google.com" in str(response.url) or response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail="Cannot download file from Google Drive. Please ensure the file is publicly accessible (Anyone with the link can view). You can make it public by clicking 'Share' → 'Change to anyone with the link' → 'Viewer'."
                        )
                else:
                    # For Google Docs/Sheets/Slides, the export failed
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot export {doc_type} from Google Drive. Please ensure the document is publicly accessible (Anyone with the link can view) and try again."
                    )
            
            # Additional check: make sure we didn't get HTML content (login page)
            content_type_header = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type_header and len(response.content) > 0:
                content_preview = response.content[:200].decode('utf-8', errors='ignore')
                if 'accounts.google.com' in content_preview or 'sign in' in content_preview.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="File requires authentication. Please make the Google Drive file publicly accessible: Share → Anyone with the link → Viewer."
                    )
            
            # Get filename and content type
            if expected_content_type and expected_extension:
                # For exported Google Docs/Sheets/Slides, use expected values
                filename = f"drive_{doc_type}_{file_id[:8]}{expected_extension}"
                content_type = expected_content_type
            else:
                # For regular files, extract from response
                filename = _extract_filename_from_response(response, file_id)
                content_type = _get_content_type_from_filename(filename)
            
            content = response.content
            
            logger.info(f"[{request_id}] Downloaded file: {filename} ({len(content)} bytes, type: {content_type})")
        
        # Validate file type
        allowed_types = [
            "application/pdf", 
            "image/png", "image/jpeg", "image/jpg",
            "text/csv", "application/csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/plain"
        ]
        
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {content_type} not supported. Supported: PDF, Images, CSV, Word, Excel, Text"
            )
        
        # Check for duplicates
        existing_docs = await search_service.list_documents()
        for doc in existing_docs:
            if doc['filename'] == filename:
                logger.info(f"[{request_id}] File already exists, skipping: {filename}")
                return DocumentUploadResponse(
                    id=doc['id'],
                    display_name=filename,
                    file_type=content_type,
                    upload_time=datetime.now(),
                    status="skipped - already exists"
                )
        
        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document using enhanced processor
            processed_doc = await enhanced_processor.process_file(
                file_path=temp_file_path,
                file_type=content_type,
                filename=filename
            )
            
            logger.info(f"[{request_id}] Enhanced processing complete: {processed_doc['chunk_count']} semantic chunks")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {
                'file_type': content_type,
                'upload_time': datetime.now().isoformat(),
                'chunk_count': processed_doc['chunk_count'],
                'char_count': processed_doc['character_count'],
                'source': 'google_drive',
                'original_url': request.drive_url
            }
            
            # Add enhanced metadata
            if 'metadata' in processed_doc:
                metadata.update({
                    'word_count': processed_doc['metadata']['word_count'],
                    'has_tables': processed_doc['metadata']['has_tables'],
                    'sections': processed_doc['metadata']['sections'][:5],
                    'entities': processed_doc['metadata']['entities']
                })
            
            # Index in Vertex AI Search
            logger.info(f"[{request_id}] Indexing in Vertex AI Search...")
            await search_service.index_document(
                document_id=document_id,
                filename=filename,
                text_content=processed_doc['full_text'],
                chunks=processed_doc['chunks'],
                metadata=metadata
            )
            
            logger.info(f"[{request_id}] Google Drive import completed successfully!")
            
            return DocumentUploadResponse(
                id=document_id,
                display_name=filename,
                file_type=content_type,
                upload_time=datetime.now(),
                status="success"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error importing from Google Drive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_drive_file_id(url: str) -> tuple:
    """Extract file ID and type from various Google Drive URL formats"""
    import re
    
    # Common Google Drive URL patterns with type detection
    patterns = [
        (r'drive\.google\.com/file/d/([a-zA-Z0-9-_]+)', 'file'),
        (r'drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)', 'file'),
        (r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)', 'document'),
        (r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)', 'spreadsheet'),
        (r'docs\.google\.com/presentation/d/([a-zA-Z0-9-_]+)', 'presentation'),
    ]
    
    for pattern, doc_type in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), doc_type
    
    return None, None


def _extract_filename_from_response(response, file_id: str) -> str:
    """Extract filename from response headers or generate one"""
    import re
    
    # Try to get filename from Content-Disposition header
    content_disposition = response.headers.get('content-disposition', '')
    if content_disposition:
        filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
        if filename_match:
            filename = filename_match.group(1).strip(' "\'')
            if filename:
                return filename
    
    # Try to get from Content-Type
    content_type = response.headers.get('content-type', '')
    extension = _get_extension_from_content_type(content_type)
    
    return f"drive_file_{file_id[:8]}{extension}"


def _get_content_type_from_filename(filename: str) -> str:
    """Determine content type from filename extension"""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    type_mapping = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'csv': 'text/csv',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'txt': 'text/plain'
    }
    
    return type_mapping.get(extension, 'application/octet-stream')


def _get_extension_from_content_type(content_type: str) -> str:
    """Get file extension from content type"""
    type_mapping = {
        'application/pdf': '.pdf',
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'text/csv': '.csv',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'text/plain': '.txt'
    }
    
    return type_mapping.get(content_type, '.bin')


# Quick test for debugging
if __name__ == "__main__":
    test_urls = [
        "https://drive.google.com/file/d/1Fudl7z8pBfPF_zLCF0Xsg0CY6-Fg0hbf/view?usp=sharing",
        "https://docs.google.com/document/d/1abc123/edit",
        "https://docs.google.com/spreadsheets/d/1xyz789/edit"
    ]
    
    for url in test_urls:
        file_id, doc_type = _extract_drive_file_id(url)
        print(f"URL: {url}")
        print(f"  File ID: {file_id}, Type: {doc_type}")
        print()