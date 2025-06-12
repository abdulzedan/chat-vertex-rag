# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev       # Development server on port 3000
npm run build     # Production build
npm run lint      # Run ESLint
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## High-Level Architecture

### RAG Engine Integration
The application uses Vertex AI RAG Engine, which is a fully managed service that:
- Automatically handles document chunking and embedding generation
- Manages vector storage internally (no external database needed)
- Provides retrieval and ranking capabilities
- Integrates seamlessly with Gemini models

Key architectural decisions:
1. **No Database Required**: Vertex AI RAG Engine manages all storage
2. **Streaming Responses**: Uses Server-Sent Events (SSE) for real-time chat
3. **Document Processing**: Automatic handling of PDFs, images, and CSVs
4. **Hybrid Approach**: For small documents (<10 pages), can use direct context instead of RAG

### Backend Structure
- **FastAPI Application**: Async Python web framework for high performance
- **Service Layer**: Separates business logic from API endpoints
  - `rag_engine.py`: Core RAG operations and corpus management
  - `document_uploader.py`: File processing for different formats
  - `gemini_client.py`: Direct Gemini model interactions
- **API Layer**: RESTful endpoints + WebSocket support
  - Document CRUD operations
  - Streaming chat/query endpoints
  - Real-time WebSocket connections

### Frontend Architecture
- **React 19 + TypeScript**: Type-safe component development
- **shadcn/ui**: Pre-built, customizable UI components
- **Tailwind CSS**: Utility-first styling
- **State Management**: React hooks for local state
- **Real-time Updates**: SSE for streaming responses

### Key Integration Points
1. **Document Upload Flow**:
   - Frontend uploads to `/api/documents/upload`
   - Backend processes file based on type
   - RAG Engine automatically chunks and embeds
   
2. **Query Flow**:
   - User question → Backend → RAG retrieval → Gemini generation
   - Responses stream back via SSE
   - Frontend updates UI in real-time

3. **Error Handling**:
   - Graceful degradation if RAG corpus unavailable
   - Fallback to direct Gemini for small documents
   - User-friendly error messages

## Important Considerations

- **GCP Credentials**: Must set GOOGLE_APPLICATION_CREDENTIALS env var
- **Corpus Initialization**: Happens automatically on first startup
- **File Size Limits**: Default 10MB, configurable in FastAPI
- **Streaming**: Requires proper proxy configuration in development