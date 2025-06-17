# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 600
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

## End State For User 
The ideal state for the application is the following:
The user will head to this browser, be able to browse documents on the page through uploading it (which would then get processed in the backend and stored in Vertex AI Search). Select the documents they want to use for chat. When the documents are selected, they are able to click on a button in the top right of the page, which would open a chat interface slide window. The users an have a chat and ask questions about the documents they selected. 


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
- **VENV** 
  - Virtual Environment for python is in the backend directory, it can be started by 'source backend/venv/bin/activate'

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

4. **Important Files**
   - These files are used to help you understand how to implement or some considerations to make:
	- rag_vertex_example.md (this is an example implementation)
	- document_processors.md (this is an example processor)



## Important Considerations

- **GCP Credentials**: Must set GOOGLE_APPLICATION_CREDENTIALS env var
- **Corpus Initialization**: Happens automatically on first startup
- **File Size Limits**: Default 10MB, configurable in FastAPI
- **Streaming**: Requires proper proxy configuration in development
