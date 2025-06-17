from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, Set

from app.services.vertex_search import VertexSearchService
from app.services.gemini_client import GeminiClient

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Set[WebSocket] = set()

# Initialize search service
search_service = VertexSearchService()

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
                        # Use Vertex AI Search + Gemini for document queries
                        search_results = await search_service.search_documents(
                            query=question,
                            max_results=10
                        )
                        
                        if search_results:
                            response_text = await search_service.generate_response(question, search_results)
                            # Simulate streaming by sending chunks
                            words = response_text.split(' ')
                            full_response = []
                            for i in range(0, len(words), 5):
                                chunk = ' '.join(words[i:i+5]) + ' '
                                await websocket.send_json({
                                    "type": "chunk",
                                    "content": chunk
                                })
                                full_response.append(chunk)
                        else:
                            # No documents found
                            response_text = "I couldn't find any relevant documents to answer your question. Please upload some documents first."
                            await websocket.send_json({
                                "type": "chunk",
                                "content": response_text
                            })
                            full_response = [response_text]
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