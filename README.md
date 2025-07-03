# RAG Demo with Vertex AI Search and Gemini

Upload documents, search across them intelligently, and chat with your documents using Google's Vertex AI Search and Gemini.

## Features

- **Multi-format support**: PDFs, images, CSV, Word, Excel, plain text
- **Intelligent search**: Multi-document queries with conversation context
- **Streaming chat**: Real-time responses with document citations
- **Smart processing**: Table-aware chunking and entity extraction

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
   GCP_PROJECT_ID=your-project-id
   GCP_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   GEMINI_MODEL=gemini-2.0-flash-001
   ```

3. **Start servers**:
   ```bash
   # Backend
   uvicorn app.main:app --reload --port 8000
   
   # Frontend
   cd frontend && npm install && npm run dev
   ```

4. **Open**: [http://localhost:3000](http://localhost:3000)

## Usage

1. **Upload documents**: Drag and drop files or click to browse
2. **Select documents**: Use checkboxes to choose which documents to query  
3. **Open chat**: Click the chat button after selecting documents
4. **Ask questions**: Type questions about your documents
5. **Get answers**: Receive streaming responses with citations

## Development

```bash
# Setup
./setup-dev.sh

# Run quality checks
pre-commit run --all-files

# Backend linting
cd backend && ruff check app/ --fix && ruff format app/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.