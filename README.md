# RAG Engine Demo

A NotebookLM-inspired document Q&A application powered by Google Cloud Vertex AI Search and Gemini 2.0 Flash. Upload documents, select which ones to search, and have intelligent conversations about their content.

## 🚀 Features

### 📄 Multi-Format Document Support
- **PDFs**: Advanced processing with Document AI Layout Parser and Gemini Vision
- **Images**: OCR text extraction with Gemini Vision capabilities  
- **CSV/Excel**: Table-aware processing with markdown formatting
- **Word Documents**: Structure-preserving extraction
- **Plain Text**: Multi-encoding support

### 🔍 Intelligent Search & Chat
- **Multi-Document Queries**: Select specific documents or search across all
- **Conversation Context**: Follow-up questions that remember previous context
- **Streaming Responses**: Real-time response generation with typing indicators
- **Rich Formatting**: Tables, citations, lists, and code blocks in responses
- **Document Citations**: Automatic source attribution and references

### 🎯 Advanced Processing
- **Hierarchical Processing**: Document AI → Gemini Multimodal → Standard libraries
- **Table-Aware Chunking**: Preserves table integrity across document chunks
- **Semantic Chunking**: Intelligent text segmentation with context preservation
- **Entity Extraction**: Automatic extraction of dates, percentages, currency, etc.

## 🛠 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Google Cloud Project with Vertex AI enabled
- Google Cloud credentials configured

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in the backend directory:

```bash
# Required
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional optimizations
USE_DOCUMENT_AI=false
USE_VERTEX_RANKING=false
USE_VERTEX_GROUNDING=false
GEMINI_MODEL=gemini-2.0-flash-001
```

### 3. Start Backend Server

```bash
uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 600
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Development server on port 3000
```

### 5. Access Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📖 Usage Guide

### Document Management
1. **Upload Documents**: Drag and drop files or click to browse
2. **View Documents**: Click on any document to preview (if supported)
3. **Select for Search**: Use checkboxes to choose which documents to query
4. **Delete Documents**: Individual or bulk deletion options

### Chat Interface  
1. **Open Chat**: Click the chat button (top-right) after selecting documents
2. **Ask Questions**: Type questions about your selected documents
3. **Follow-up Questions**: Ask related questions that build on previous responses
4. **Copy Responses**: Click copy button on any response
5. **Clear Chat**: Reset conversation context when needed

### Example Queries
- "What are the main points in this document?"
- "Compare the pricing between these two documents"
- "What percentage increases are mentioned?"
- "Summarize the key findings from the research"

## 🔧 Configuration Options

### Document AI Enhancement
Enable Document AI for better PDF processing:
```bash
USE_DOCUMENT_AI=true
```

### Response Quality Improvements
```bash
USE_VERTEX_RANKING=true      # Better result ranking
USE_VERTEX_GROUNDING=true    # Response validation
```

### Model Selection
```bash
GEMINI_MODEL=gemini-2.0-flash-001    # Default (fastest)
GEMINI_MODEL=gemini-1.5-pro          # Higher quality
```

## 📊 Supported File Types

| Format | Extensions | Processing Method |
|--------|------------|-------------------|
| PDF | `.pdf` | Document AI → Gemini Vision → PyPDF2 |
| Images | `.png`, `.jpg`, `.jpeg` | Gemini Vision → Standard OCR |
| CSV | `.csv` | Enhanced parser with table formatting |
| Word | `.docx` | Structure-preserving extraction |
| Excel | `.xlsx` | Multi-sheet processing |
| Text | `.txt` | Multi-encoding support |

## 🏗 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI Backend│    │  Google Cloud   │
│                 │    │                  │    │                 │
│ • Document List │────│ • Upload API     │────│ • Vertex AI     │
│ • Chat Interface│    │ • Search API     │    │ • Gemini 2.0    │
│ • File Upload   │    │ • Streaming      │    │ • Document AI   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧪 Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests  
cd frontend
npm test
```

### Building for Production

```bash
# Frontend build
cd frontend
npm run build

# Backend dependencies
cd backend
pip install -r requirements.txt
```

### Code Quality

```bash
# Frontend linting
cd frontend
npm run lint

# Backend formatting (if configured)
cd backend
ruff check app/
```

## 🚨 Troubleshooting

### Common Issues

**"No documents found"**
- Check that documents are successfully uploaded and indexed
- Verify Google Cloud credentials are properly configured
- Ensure Vertex AI Search data store is created

**Slow document processing**
- Enable Document AI for better PDF processing
- Check file sizes (20MB limit for Gemini processing)
- Monitor backend logs for processing method used

**Streaming not working**
- Verify proxy configuration in frontend for SSE
- Check network configuration allows Server-Sent Events
- Ensure backend streaming headers are properly set

**Authentication errors**
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account
- Ensure service account has Vertex AI permissions
- Check that Vertex AI APIs are enabled in your GCP project

### Log Analysis

Backend logs show processing hierarchy:
```
INFO: Processing PDF with Document AI Layout Parser    # Best quality
INFO: Trying Gemini multimodal as fallback           # Fast fallback  
INFO: Falling back to PyPDF2 standard processing     # Basic fallback
```

## 📝 API Documentation

### Document Endpoints
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents  
- `DELETE /api/documents/{id}` - Delete specific document

### Chat Endpoints
- `POST /api/chat/query` - Streaming query with SSE
- `GET /api/conversations/{session_id}` - Get conversation history
- `DELETE /api/conversations/{session_id}` - Clear conversation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- 📧 Email: [your-email@example.com]
- 💬 GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- 📖 Documentation: See `TECHNICAL_GUIDE.md` for detailed technical information

## 🙏 Acknowledgments

- Built with Google Cloud Vertex AI and Gemini
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- Inspired by Google's NotebookLM