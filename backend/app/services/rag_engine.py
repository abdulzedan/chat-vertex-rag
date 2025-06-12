import os
from typing import Optional, List, AsyncGenerator
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import logging

logger = logging.getLogger(__name__)

# Global variable to store corpus
_rag_corpus: Optional[rag.RagCorpus] = None

async def init_rag_corpus() -> rag.RagCorpus:
    """Initialize or get existing RAG corpus"""
    global _rag_corpus
    
    if _rag_corpus:
        return _rag_corpus
    
    try:
        # Try to get existing corpus first
        corpora = rag.list_corpora()
        for corpus in corpora:
            if corpus.display_name == "notebooklm_demo":
                _rag_corpus = corpus
                logger.info(f"Using existing RAG corpus: {corpus.name}")
                return _rag_corpus
        
        # Create new corpus if not found
        _rag_corpus = rag.create_corpus(display_name="notebooklm_demo")
        logger.info(f"Created new RAG corpus: {_rag_corpus.name}")
        return _rag_corpus
        
    except Exception as e:
        logger.error(f"Error initializing RAG corpus: {e}")
        raise

def get_rag_corpus() -> rag.RagCorpus:
    """Get the initialized RAG corpus"""
    if not _rag_corpus:
        raise RuntimeError("RAG corpus not initialized. Call init_rag_corpus() first.")
    return _rag_corpus

async def upload_document(
    file_path: str, 
    display_name: str,
    description: Optional[str] = None
) -> rag.RagFile:
    """Upload a document to the RAG corpus"""
    import asyncio
    from google.cloud import storage
    corpus = get_rag_corpus()
    
    try:
        logger.info(f"Starting upload for file: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")
        logger.info(f"Corpus name: {corpus.name}")
        
        # For development/testing, try direct upload_file first (faster for small files)
        file_size = os.path.getsize(file_path)
        
        if file_size < 10 * 1024 * 1024:  # Files under 10MB - try direct upload first
            logger.info(f"Small file ({file_size} bytes), trying direct upload_file first...")
            try:
                loop = asyncio.get_event_loop()
                
                def direct_upload():
                    return rag.upload_file(
                        corpus_name=corpus.name,
                        path=file_path,
                        display_name=display_name,
                        description=description
                    )
                
                # Try direct upload with short timeout
                rag_file = await asyncio.wait_for(
                    loop.run_in_executor(None, direct_upload),
                    timeout=45.0  # 45 second timeout for direct upload
                )
                
                logger.info(f"Direct upload completed! File: {display_name} with ID: {rag_file.name}")
                return rag_file
                
            except asyncio.TimeoutError:
                logger.warning(f"Direct upload timed out after 45s, falling back to GCS method...")
            except Exception as e:
                logger.warning(f"Direct upload failed: {e}, falling back to GCS method...")
        
        # Fall back to GCS + import_files method
        logger.info("Using GCS + import_files method...")
        import uuid
        project_id = os.getenv("GCP_PROJECT_ID")
        bucket_name = f"{project_id}-rag-temp"
        # Add unique ID to prevent conflicts
        unique_id = str(uuid.uuid4())[:8]
        blob_name = f"uploads/{unique_id}_{display_name}"
        gcs_path = f"gs://{bucket_name}/{blob_name}"
        
        logger.info(f"Uploading to GCS: {gcs_path}")
        
        try:
            # Create bucket if it doesn't exist
            storage_client = storage.Client()
            try:
                bucket = storage_client.get_bucket(bucket_name)
            except:
                logger.info(f"Creating bucket: {bucket_name}")
                bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            
            # Upload file to GCS
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            logger.info(f"Uploaded to GCS: {gcs_path}")
            
            # Start import but don't wait
            logger.info("Starting background import from GCS...")
            loop = asyncio.get_event_loop()
            
            def start_import():
                return rag.import_files(
                    corpus_name=corpus.name,
                    paths=[gcs_path],
                    transformation_config=rag.TransformationConfig(
                        chunking_config=rag.ChunkingConfig(
                            chunk_size=512,
                            chunk_overlap=100
                        )
                    ),
                    max_embedding_requests_per_min=1000
                )
            
            # Start import in background - don't wait
            loop.run_in_executor(None, start_import)
            logger.info("Import started in background")
            
            # Return placeholder
            class ImportingRagFile:
                def __init__(self, name, display_name):
                    self.name = f"importing_{unique_id}_{display_name}"
                    self.display_name = display_name
            
            return ImportingRagFile(f"importing_{display_name}", display_name)
            
        except Exception as e:
            logger.error(f"Error with GCS upload/import: {e}")
            
            # Check if it's a concurrent operation error
            if "There are other operations running" in str(e):
                logger.warning("Another import operation is already running. Waiting for it to complete...")
                
                # Extract operation ID from error message
                import re
                match = re.search(r'Operation IDs are: \[(\d+)\]', str(e))
                if match:
                    operation_id = match.group(1)
                    logger.info(f"Waiting for operation {operation_id} to complete...")
                
                # Wait longer and poll for the file
                import asyncio
                max_attempts = 30  # 5 minutes total
                for attempt in range(max_attempts):
                    await asyncio.sleep(10)  # Check every 10 seconds
                    logger.info(f"Checking if file has been imported (attempt {attempt + 1}/{max_attempts})...")
                    
                    files = await list_documents()
                    for file in files:
                        if display_name in file.display_name:
                            logger.info(f"File was imported by concurrent operation: {file.name}")
                            # Clean up our GCS file since import succeeded
                            try:
                                blob.delete()
                                logger.info("Cleaned up temporary GCS file")
                            except:
                                pass
                            return file
                
                # If still not found after waiting, the operation might have failed
                logger.warning("File not found after waiting. The import operation may have failed.")
            
            # Fall back to direct upload_file for small files
            if os.path.getsize(file_path) < 25 * 1024 * 1024:  # Under 25MB limit
                logger.info("Falling back to direct upload_file...")
                return await _upload_small_file(file_path, display_name, description, corpus)
            else:
                raise
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise

async def import_from_drive(drive_url: str) -> str:
    """Import documents from Google Drive URL"""
    corpus = get_rag_corpus()
    
    try:
        logger.info(f"Starting Google Drive import from: {drive_url}")
        
        # Extract display name from URL
        if "/file/d/" in drive_url:
            display_name = f"Google Drive File"
        elif "/drive/folders/" in drive_url:
            display_name = f"Google Drive Folder"
        else:
            display_name = "Google Drive Import"
        
        # Run import_files for Google Drive
        import asyncio
        loop = asyncio.get_event_loop()
        
        def drive_import():
            try:
                logger.info(f"Calling import_files with:")
                logger.info(f"  - corpus: {corpus.name}")
                logger.info(f"  - drive_url: {drive_url}")
                
                response = rag.import_files(
                    corpus_name=corpus.name,
                    paths=[drive_url],
                    transformation_config=rag.TransformationConfig(
                        chunking_config=rag.ChunkingConfig(
                            chunk_size=512,
                            chunk_overlap=100
                        )
                    ),
                    max_embedding_requests_per_min=1000
                )
                
                logger.info(f"Import response: {response}")
                return response
                
            except Exception as e:
                logger.error(f"Error in drive_import function: {e}")
                logger.error(f"Error type: {type(e)}")
                
                # Check if it's a concurrent operation error
                if "There are other operations running" in str(e):
                    logger.info("Concurrent operation detected - another import is already running")
                    # Extract operation ID from error message
                    import re
                    match = re.search(r'Operation IDs are: \[(\d+)\]', str(e))
                    if match:
                        operation_id = match.group(1)
                        logger.info(f"Existing operation ID: {operation_id}")
                    raise RuntimeError(f"Another import operation is already running. Please wait for it to complete before trying again.")
                else:
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
        
        # Try import with longer timeout and better error handling
        try:
            import_response = await asyncio.wait_for(
                loop.run_in_executor(None, drive_import),
                timeout=120.0  # 2 minute timeout to start the operation
            )
            
            logger.info(f"Google Drive import completed! Response: {import_response}")
            logger.info(f"Response type: {type(import_response)}")
            logger.info(f"Response attributes: {dir(import_response)}")
            
            if hasattr(import_response, 'imported_rag_files_count'):
                logger.info(f"Imported {import_response.imported_rag_files_count} files from Drive")
            if hasattr(import_response, 'skipped_rag_files_count'):
                logger.info(f"Skipped {import_response.skipped_rag_files_count} files (already exist)")
            if hasattr(import_response, 'partial_failures'):
                logger.info(f"Partial failures: {import_response.partial_failures}")
            
            return f"drive_import_{drive_url.split('/')[-1]}"
            
        except asyncio.TimeoutError:
            logger.warning("Drive import operation timed out after 2 minutes. It may still be processing in background.")
            # Check if any files appeared during the wait
            try:
                files = await list_documents()
                logger.info(f"Current file count in corpus: {len(files)}")
                for file in files:
                    logger.info(f"File in corpus: {file.display_name}")
            except Exception as check_error:
                logger.error(f"Error checking files after timeout: {check_error}")
            
            return f"importing_drive_{drive_url.split('/')[-1]}"
        
    except Exception as e:
        logger.error(f"Error importing from Google Drive: {e}")
        raise Exception(f"Failed to import from Google Drive: {str(e)}")

async def _upload_small_file(file_path: str, display_name: str, description: Optional[str], corpus) -> rag.RagFile:
    """Fallback method for small files using direct upload_file"""
    import asyncio
    
    logger.info("Using direct upload_file method (fallback for small files)...")
    loop = asyncio.get_event_loop()
    
    def simple_upload():
        return rag.upload_file(
            corpus_name=corpus.name,
            path=file_path,
            display_name=display_name,
            description=description
        )
    
    try:
        upload_task = loop.run_in_executor(None, simple_upload)
        logger.info("Waiting for direct upload to complete (timeout: 30s)...")
        rag_file = await asyncio.wait_for(upload_task, timeout=30.0)  # 30 second timeout for small files
        logger.info(f"Direct upload completed! File: {display_name} with ID: {rag_file.name}")
        return rag_file
    except asyncio.TimeoutError:
        logger.error("Direct upload timed out")
        # Return a temporary file object for the UI
        class TempRagFile:
            def __init__(self, name, display_name):
                self.name = f"temp_{name}"
                self.display_name = display_name
        
        return TempRagFile(display_name, display_name)

async def list_documents() -> List[rag.RagFile]:
    """List all documents in the RAG corpus"""
    corpus = get_rag_corpus()
    
    try:
        # Use the correct list_files method from documentation
        files = rag.list_files(corpus_name=corpus.name)
        return list(files)
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise

async def delete_document(file_name: str) -> bool:
    """Delete a document from the RAG corpus"""
    try:
        # Use the correct delete_file method from documentation
        rag.delete_file(name=file_name)
        logger.info(f"Deleted file: {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise

async def query_documents(
    question: str,
    similarity_top_k: int = 10,
    vector_distance_threshold: float = 0.5
) -> AsyncGenerator[str, None]:
    """Query documents using RAG and stream the response"""
    corpus = get_rag_corpus()
    
    logger.info(f"Starting RAG query: '{question}'")
    
    # First check if there are any files in the corpus
    try:
        files = await list_documents()
        logger.info(f"Found {len(files)} files in RAG corpus")
        
        if len(files) == 0:
            logger.warning("No files found in RAG corpus. Documents may still be importing.")
            yield "I notice that there are currently no documents available in the knowledge base. "
            yield "This might be because documents are still being processed and imported. "
            yield "Please wait a few minutes for the document import to complete, then try your question again. "
            yield "\n\nIf you just uploaded a document, the import process can take 2-5 minutes depending on the file size."
            return
            
        # Log which files are available
        for file in files:
            logger.info(f"Available document: {file.display_name}")
            
    except Exception as e:
        logger.error(f"Error checking files in corpus: {e}")
    
    # Create retrieval tool following the documentation pattern
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus.name,
                    )
                ],
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=similarity_top_k,
                    filter=rag.utils.resources.Filter(
                        vector_distance_threshold=vector_distance_threshold
                    ),
                ),
            ),
        )
    )
    
    # Create model with RAG tool
    model = GenerativeModel(
        model_name="gemini-2.0-flash-001",  # Use the specific version
        tools=[rag_retrieval_tool]
    )
    
    logger.info("Generating RAG response...")
    
    try:
        # Generate response with streaming
        response_stream = model.generate_content(
            question,
            stream=True
        )
        
        # Stream the response
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        yield f"I apologize, but I encountered an error while searching the documents: {str(e)}"

async def query_direct(
    document_content: str,
    question: str
) -> AsyncGenerator[str, None]:
    """Query a document directly without RAG (for small documents)"""
    model = GenerativeModel("gemini-2.0-flash-exp")
    
    prompt = f"""Based on the following document, please answer the question.

Document:
{document_content}

Question: {question}

Answer:"""
    
    response_stream = model.generate_content(
        prompt,
        stream=True
    )
    
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text