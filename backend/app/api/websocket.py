import json
import logging
from datetime import datetime
from typing import Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.gemini_client import GeminiClient
from app.services.vertex_search import VertexSearchService

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Set[WebSocket] = set()

# Store activity log connections (separate from chat)
activity_connections: Set[WebSocket] = set()

# Initialize search service
search_service = VertexSearchService()


# =============================================================================
# Activity Broadcaster — pushes pipeline events to connected frontend clients
# =============================================================================

class ActivityBroadcaster:
    """Broadcasts structured activity events to WebSocket clients.

    Events have: timestamp, stage, message, detail (optional), type (info/success/warning/error).
    """

    async def emit(
        self,
        stage: str,
        message: str,
        detail: Optional[str] = None,
        event_type: str = "info",
    ):
        """Broadcast an activity event to all connected activity clients."""
        event = {
            "type": "activity",
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "event_type": event_type,
        }
        if detail:
            event["detail"] = detail

        disconnected = set()
        for ws in activity_connections:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.add(ws)

        # Clean up disconnected clients
        activity_connections.difference_update(disconnected)

    async def emit_start(self, stage: str, message: str):
        """Emit an in-progress event."""
        await self.emit(stage, message, event_type="info")

    async def emit_success(self, stage: str, message: str, detail: Optional[str] = None):
        """Emit a success event."""
        await self.emit(stage, message, detail=detail, event_type="success")

    async def emit_warning(self, stage: str, message: str, detail: Optional[str] = None):
        """Emit a warning event."""
        await self.emit(stage, message, detail=detail, event_type="warning")

    async def emit_error(self, stage: str, message: str, detail: Optional[str] = None):
        """Emit an error event."""
        await self.emit(stage, message, detail=detail, event_type="error")

    async def clear(self):
        """Send a clear signal to reset the activity log on clients."""
        event = {"type": "activity_clear", "timestamp": datetime.now().isoformat()}
        disconnected = set()
        for ws in activity_connections:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.add(ws)
        activity_connections.difference_update(disconnected)


# Global broadcaster instance — imported by other modules
activity_broadcaster = ActivityBroadcaster()


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@router.websocket("/activity")
async def activity_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time activity log events."""
    await websocket.accept()
    activity_connections.add(websocket)
    logger.info("Activity WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        activity_connections.discard(websocket)
        logger.info("Activity WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Activity WebSocket error: {e}")
        activity_connections.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


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
                await websocket.send_json({"type": "status", "status": "processing"})

                try:
                    if use_rag:
                        # Use Vertex AI Search + Gemini for document queries
                        search_results = await search_service.search_documents(
                            query=question, max_results=10
                        )

                        if search_results:
                            response_text = await search_service.generate_response(
                                question, search_results
                            )
                            # Simulate streaming by sending chunks
                            words = response_text.split(" ")
                            full_response = []
                            for i in range(0, len(words), 5):
                                chunk = " ".join(words[i : i + 5]) + " "
                                await websocket.send_json(
                                    {"type": "chunk", "content": chunk}
                                )
                                full_response.append(chunk)
                        else:
                            # No documents found
                            response_text = "I couldn't find any relevant documents to answer your question. Please upload some documents first."
                            await websocket.send_json(
                                {"type": "chunk", "content": response_text}
                            )
                            full_response = [response_text]
                    else:
                        # Use direct Gemini
                        gemini_client = GeminiClient()
                        response_generator = gemini_client.generate_stream(question)

                        # Stream response chunks
                        full_response = []
                        async for chunk in response_generator:
                            await websocket.send_json(
                                {"type": "chunk", "content": chunk}
                            )
                            full_response.append(chunk)

                    # Send completion signal
                    await websocket.send_json(
                        {"type": "complete", "full_response": "".join(full_response)}
                    )

                except Exception as e:
                    await websocket.send_json({"type": "error", "error": str(e)})

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.remove(websocket)
        await websocket.close()
