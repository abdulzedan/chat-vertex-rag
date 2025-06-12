from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import logging

from app.models.schemas import ChatRequest, QueryRequest
from app.services.rag_engine import query_documents, query_direct
from app.services.gemini_client import GeminiClient

router = APIRouter()
logger = logging.getLogger(__name__)

async def stream_response(generator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    """Convert string generator to SSE format"""
    try:
        async for chunk in generator:
            # Format as Server-Sent Events
            data = json.dumps({"content": chunk})
            yield f"data: {data}\n\n".encode()
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n".encode()
    finally:
        # Send done signal
        yield f"data: {json.dumps({'done': True})}\n\n".encode()

@router.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query documents using RAG with streaming response"""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Use RAG query for document-based questions
        try:
            logger.info("Attempting RAG query...")
            response_generator = query_documents(
                question=request.question,
                similarity_top_k=request.similarity_top_k,
                vector_distance_threshold=request.vector_distance_threshold
            )
        except Exception as rag_error:
            logger.error(f"RAG query failed: {rag_error}")
            # Fall back to direct Gemini
            gemini_client = GeminiClient()
            response_generator = gemini_client.generate_stream(
                prompt=f"You are a helpful AI assistant. The user is asking: {request.question}. Please provide a helpful response.",
                system_instruction="You are a helpful assistant."
            )
        
        # Return streaming response
        return StreamingResponse(
            stream_response(response_generator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with optional RAG"""
    try:
        if request.use_rag:
            # Use RAG for document-based queries
            response_generator = query_documents(
                question=request.question,
                similarity_top_k=10,
                vector_distance_threshold=0.5
            )
        else:
            # Use direct Gemini for general queries
            gemini_client = GeminiClient()
            response_generator = gemini_client.generate_stream(
                prompt=request.question,
                system_instruction="You are a helpful assistant answering questions about documents."
            )
        
        return StreamingResponse(
            stream_response(response_generator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))