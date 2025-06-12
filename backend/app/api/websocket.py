from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, Set

from app.services.rag_engine import query_documents
from app.services.gemini_client import GeminiClient

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Set[WebSocket] = set()

@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "chat":
                question = message.get("question", "")
                use_rag = message.get("use_rag", True)
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "status": "processing"
                })
                
                try:
                    if use_rag:
                        # Use RAG for document queries
                        response_generator = query_documents(question)
                    else:
                        # Use direct Gemini
                        gemini_client = GeminiClient()
                        response_generator = gemini_client.generate_stream(question)
                    
                    # Stream response chunks
                    full_response = []
                    async for chunk in response_generator:
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk
                        })
                        full_response.append(chunk)
                    
                    # Send completion signal
                    await websocket.send_json({
                        "type": "complete",
                        "full_response": "".join(full_response)
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.remove(websocket)
        await websocket.close()