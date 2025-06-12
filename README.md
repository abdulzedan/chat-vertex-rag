# NotebookLM RAG Demo

A proof-of-concept RAG (Retrieval Augmented Generation) application using Google Cloud Platform's Vertex AI RAG Engine and Gemini 2.0 Flash. This demo replicates Google's NotebookLM experience for document Q&A.

## Features

- **Document Upload**: Support for PDFs, images (PNG, JPG), and CSV files
- **Automatic Processing**: Vertex AI RAG Engine handles chunking and embedding automatically
- **Real-time Chat**: Ask questions about your documents with streaming responses
- **Modern UI**: Clean interface with document list, viewer, and slide-out chat
- **Fully Managed**: No database setup required - uses Vertex AI's managed RAG service

## Prerequisites

- Python 3.9+
- Node.js 18+
- Google Cloud Project with billing enabled
- Service account with Vertex AI permissions

## Setup Instructions

### 1. Enable GCP APIs

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Set up Authentication

```bash
# For local development, simply run:
gcloud auth application-default login

# Make sure you're using the correct project
gcloud config set project YOUR_PROJECT_ID
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your values:
# GCP_PROJECT_ID=your-project-id
# GCP_LOCATION=us-central1

# Run the backend
uvicorn app.main:app --reload --port 8000
```

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The application will be available at http://localhost:3000

## Usage

1. **Upload Documents**: Click "Upload Document" in the left sidebar and drag & drop or select files
2. **View Documents**: Click on any document in the list to select it
3. **Ask Questions**: Click the chat button (bottom right) to open the Q&A interface
4. **Get Answers**: Type your question and get AI-powered answers based on your documents

## Architecture

### Backend (FastAPI)
- `app/services/rag_engine.py`: Vertex AI RAG Engine integration
- `app/services/document_uploader.py`: Document processing logic
- `app/api/documents.py`: Document management endpoints
- `app/api/chat.py`: Query and chat endpoints with SSE streaming

### Frontend (React + TypeScript)
- `components/DocumentList.tsx`: Document management interface
- `components/DocumentViewer.tsx`: Document preview placeholder
- `components/ChatInterface.tsx`: Real-time Q&A chat
- `components/FileUploader.tsx`: Drag-and-drop file upload

## Performance Notes

- For <100 documents, the RAG Engine provides excellent performance
- Initial document processing typically takes ~10 seconds
- Query responses stream in real-time
- For very small document sets (<10 docs), consider using direct context instead of RAG

## Troubleshooting

### Common Issues

1. **"RAG corpus not initialized"**
   - The RAG corpus is created automatically on first startup
   - Check your GCP credentials and project permissions

2. **Upload failures**
   - Ensure the backend is running on port 8000
   - Check file size limits (default: 10MB)
   - Verify supported file types: PDF, PNG, JPG, CSV

3. **No streaming responses**
   - Check browser console for WebSocket errors
   - Ensure your proxy configuration is correct in vite.config.ts

## Development

### Adding New File Types

1. Update `allowed_types` in `backend/app/api/documents.py`
2. Add processing logic in `backend/app/services/document_uploader.py`
3. Update frontend file acceptance in `FileUploader.tsx`

### Customizing the UI

The frontend uses shadcn/ui components with Tailwind CSS. To add new components:

```bash
npx shadcn@latest add [component-name]
```

## Production Considerations

1. **Authentication**: Add Google Cloud IAM or custom auth
2. **Rate Limiting**: Implement API rate limits
3. **Monitoring**: Set up Cloud Logging and Monitoring
4. **Scaling**: Deploy backend to Cloud Run, frontend to Cloud CDN
5. **Security**: Enable CORS restrictions, validate file uploads

## License

This is a demo project for educational purposes.