# RAG Demo with Vertex AI Search and Gemini

A Retrieval-Augmented Generation (RAG) application that enables intelligent document search and question-answering. Upload documents, search across them intelligently, and chat with your documents using Google's Vertex AI Search and Gemini.

## Features

- **Multi-format support**: PDFs, images, CSV, Word, Excel, plain text
- **Intelligent search**: Multi-document queries with conversation context  
- **Streaming chat**: Real-time responses with document citations
- **Smart processing**: Table-aware chunking, semantic segmentation, entity extraction
- **Hierarchical processing**: Document AI → Gemini Vision → Standard libraries fallback

## Quick Start

### Prerequisites
- Python 3.9+, Node.js 16+
- Google Cloud Project with Vertex AI enabled

### Setup

1. **Backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment** (create `backend/.env`):
   ```bash
   # Required
   GCP_PROJECT_ID=your-project-id
   GCP_LOCATION=us-central1
   GEMINI_MODEL=gemini-2.0-flash-001
   
   # Vertex AI Search (created by setup-gcp.sh)
   VERTEX_SEARCH_DATASTORE_ID=rag-demo-datastore
   VERTEX_SEARCH_APP_ID=rag-demo-app
   
   # Optional (set to true for better quality)
   USE_DOCUMENT_AI=false       # Advanced PDF processing
   USE_VERTEX_RANKING=false    # Better search ranking
   USE_VERTEX_GROUNDING=false  # Response validation
   ```

3. **Google Cloud Setup**:
   ```bash
   # Setup GCP resources (run once)
   ./scripts/setup-gcp.sh
   
   # Setup authentication
   gcloud auth application-default login
   ```

4. **Start servers**:
   ```bash
   # Backend
   uvicorn app.main:app --reload --port 8000
   
   # Frontend
   cd frontend && npm install && npm run dev
   ```

5. **Open**: [http://localhost:3000](http://localhost:3000)

## Supported File Types

| Format | Extensions | Processing Method |
|--------|------------|-------------------|
| PDF | `.pdf` | Document AI → Gemini Vision → PyPDF2 |
| Images | `.png`, `.jpg`, `.jpeg` | Gemini Vision → Standard OCR |
| CSV | `.csv` | Enhanced parser with table formatting |
| Word | `.docx` | Structure-preserving extraction |
| Excel | `.xlsx` | Multi-sheet processing |
| Text | `.txt` | Multi-encoding support |

## How It Works

### Document Processing Pipeline
```
Upload → Document Processing → Chunking → Metadata Extraction → Index in Vertex AI Search
```

1. **Document Upload**: Files are processed using hierarchical fallback (Document AI → Gemini Vision → Standard libraries)
2. **Smart Chunking**: Semantic chunking with table preservation (300-1500 chars, 150 char overlap)
3. **Metadata Enhancement**: Extract entities (dates, percentages, currency) and semantic flags
4. **Vertex AI Indexing**: Automatic embeddings and enterprise search capabilities

### Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI Backend│    │ Vertex AI Search│
│ • File Upload   │────│ • Document AI    │────│ • Auto Embedding│
│ • Document List │    │ • Gemini Vision  │    │ • Vector Storage│
│ • Chat Interface│    │ • Smart Chunking │    │ • Semantic Search│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Usage

1. **Upload documents**: Drag and drop files or click to browse
2. **Select documents**: Use checkboxes to choose which documents to query  
3. **Open chat**: Click the chat button after selecting documents
4. **Ask questions**: Type questions about your documents
5. **Get answers**: Receive streaming responses with citations

### Example Queries
- "What are the main points in this document?"
- "Compare the pricing between these two documents"  
- "What percentage increases are mentioned?"
- "Summarize the key findings from the research"

## Development

### Quick Setup
```bash
# Automated setup (recommended)
./scripts/setup-dev.sh

# Manual setup
cd backend && python -m venv venv && source venv/bin/activate && pip install -e ".[dev]"
cd frontend && npm install
pre-commit install
```

### Code Quality
```bash
# Run all quality checks
pre-commit run --all-files

# Backend linting & formatting  
cd backend && source venv/bin/activate
ruff check app/ --fix && ruff format app/ && isort app/

# Frontend linting
cd frontend && npm run lint && npm run format
```

### Key Files
- `backend/app/main.py` - FastAPI server entry point
- `backend/app/services/vertex_search.py` - Vertex AI Search integration
- `backend/app/services/enhanced_document_processor.py` - Document processing pipeline
- `frontend/src/components/ChatInterface.tsx` - Main chat component
- `frontend/src/components/DocumentManager.tsx` - Document upload/management

## API Endpoints

### Document Management
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents  
- `DELETE /api/documents/{id}` - Delete specific document

### Chat Interface
- `POST /api/chat/query` - Streaming query with Server-Sent Events
- `GET /api/conversations/{session_id}` - Get conversation history
- `DELETE /api/conversations/{session_id}` - Clear conversation

## Troubleshooting

**"No documents found"**
- Verify Google Cloud credentials and Vertex AI Search data store creation

**Authentication errors**  
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account
- Check that Vertex AI APIs are enabled in your GCP project

**Slow processing**
- Enable Document AI for better PDF processing (`USE_DOCUMENT_AI=true`)
- Check file sizes (20MB limit for Gemini processing)

## License

MIT License - see [LICENSE](LICENSE) file for details.