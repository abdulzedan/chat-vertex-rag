# RAG Demo with Vertex AI Search and Gemini

A powerful, production-ready Retrieval-Augmented Generation (RAG) application that enables intelligent document search and question-answering. Built with Google Cloud's Vertex AI Search and Gemini models, this demo showcases enterprise-grade document processing and conversational AI capabilities.

## Overview

This application provides a complete solution for building document-based AI applications. Upload documents in various formats, search across them intelligently, and have natural conversations about their content - all powered by Google's latest AI technologies.

## Features

### Multi-Format Document Support
- **PDFs**: Advanced processing with Document AI Layout Parser and Gemini Vision
- **Images**: OCR text extraction with Gemini Vision capabilities  
- **CSV/Excel**: Table-aware processing with markdown formatting
- **Word Documents**: Structure-preserving extraction
- **Plain Text**: Multi-encoding support

### Intelligent Search & Chat
- **Multi-Document Queries**: Select specific documents or search across all
- **Conversation Context**: Follow-up questions that remember previous context
- **Streaming Responses**: Real-time response generation with typing indicators
- **Rich Formatting**: Tables, citations, lists, and code blocks in responses
- **Document Citations**: Automatic source attribution and references

### Advanced Processing
- **Hierarchical Processing**: Document AI → Gemini Multimodal → Standard libraries
- **Table-Aware Chunking**: Preserves table integrity across document chunks
- **Semantic Chunking**: Intelligent text segmentation with context preservation
- **Entity Extraction**: Automatic extraction of dates, percentages, currency, etc.

## Quick Start

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
# Required - Your Google Cloud configuration
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Model configuration
GEMINI_MODEL=gemini-2.0-flash-001
FALLBACK_MODELS=gemini-1.5-flash,gemini-1.5-pro

# Optional feature flags (set to true for better quality)
USE_DOCUMENT_AI=false       # Enable for advanced PDF processing
USE_VERTEX_RANKING=false    # Enable for better search ranking
USE_VERTEX_GROUNDING=false  # Enable for response validation

# Document AI configuration (if USE_DOCUMENT_AI=true)
DOCAI_LOCATION=us
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

## Usage Guide

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

## Configuration Options

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

## Supported File Types

| Format | Extensions | Processing Method |
|--------|------------|-------------------|
| PDF | `.pdf` | Document AI → Gemini Vision → PyPDF2 |
| Images | `.png`, `.jpg`, `.jpeg` | Gemini Vision → Standard OCR |
| CSV | `.csv` | Enhanced parser with table formatting |
| Word | `.docx` | Structure-preserving extraction |
| Excel | `.xlsx` | Multi-sheet processing |
| Text | `.txt` | Multi-encoding support |

## Document Processing & Indexing Pipeline

### How Documents Are Loaded into Vertex AI Search

The application uses a sophisticated multi-stage pipeline to process and index documents:

#### 1. Document Upload & Processing
```
Upload File → Document Processing → Chunking → Metadata Extraction → Index in Vertex AI Search
```

**Processing Hierarchy (Fallback Chain):**
- **Document AI** (if enabled): Best quality for PDFs with tables and complex layouts
- **Gemini Vision**: Fast multimodal processing for PDFs and images
- **Standard Libraries**: PyPDF2, python-docx, pandas for basic extraction

#### 2. Intelligent Chunking Strategy
- **Semantic Chunking**: Preserves context by breaking at natural boundaries (sentences, paragraphs)
- **Table-Aware**: Never splits tables across chunks
- **Adaptive Sizing**: 
  - Standard text: 300-1500 characters
  - Structured content (tables/lists): Up to 2000 characters
  - Overlap: 150 characters for context preservation

#### 3. Metadata Enhancement
Each chunk is enriched with structured metadata:
```json
{
  "document_type": "chunk",
  "parent_document_id": "doc_123",
  "chunk_type": "pricing|table|section_header",
  "content": "chunk text content",
  "filename": "document.pdf",
  "has_pricing_info": true,
  "has_rates": false,
  "entities": ["percentages", "currency", "dates"]
}
```

#### 4. Vertex AI Search Integration
- **Automatic Embeddings**: Vertex AI generates vector embeddings automatically
- **Managed Storage**: No external vector database required
- **Enterprise Search**: Built-in ranking, spell correction, and query expansion
- **Multi-Document Filtering**: Efficient pre-filtering by document selection

### Supported Document Sources

| Format | Processing Method | Vertex AI Integration |
|--------|-------------------|----------------------|
| **PDF** | Document AI Layout Parser → Gemini Vision → PyPDF2 | Chunked with table preservation |
| **Images** | Gemini Vision OCR → Standard OCR | Text extraction with spatial context |
| **CSV/Excel** | Enhanced pandas parser | Table-aware chunking with headers |
| **Word Documents** | Structure-preserving python-docx | Maintains formatting and hierarchy |
| **Plain Text** | Multi-encoding detection | Semantic boundary detection |

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI Backend│    │ Vertex AI Search│
│                 │    │                  │    │                 │
│ • File Upload   │────│ • Document AI    │────│ • Auto Embedding│
│ • Document List │    │ • Gemini Vision  │    │ • Vector Storage│
│ • Chat Interface│    │ • Smart Chunking │    │ • Semantic Search│
│ • Multi-Select  │    │ • Metadata Extract│   │ • Conversation  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              └─── Streaming Chat ──────┘
```

## Development

### Quick Setup

```bash
# Automated setup (recommended)
./setup-dev.sh

# Manual setup
cd backend && python -m venv venv && source venv/bin/activate && pip install -e ".[dev]" && cd ..
cd frontend && npm install && cd ..
pre-commit install && pre-commit install --hook-type commit-msg
```

### Running the Application

```bash
# Backend (from project root)
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Frontend (from project root)  
cd frontend && npm run dev
```

### Code Quality & Linting

The project uses pre-commit hooks for automatic code quality checks:

```bash
# Run all quality checks
pre-commit run --all-files

# Backend linting & formatting
cd backend && source venv/bin/activate
ruff check app/ --fix          # Lint and auto-fix
ruff format app/               # Format code
black app/                     # Alternative formatter
isort app/                     # Sort imports
mypy app/                      # Type checking
bandit -r app/                 # Security scan

# Frontend linting & formatting
cd frontend
npm run lint                   # Lint and auto-fix
npm run lint:check            # Check only (no fixes)
npm run format                # Format with Prettier
npm run format:check          # Check formatting
npm run type-check            # TypeScript type checking
```

### Running Tests

```bash
# Backend tests
cd backend && source venv/bin/activate && pytest

# Frontend tests  
cd frontend && npm test
```

### Building for Production

```bash
# Frontend build
cd frontend && npm run build

# Backend production setup
cd backend && pip install -r requirements.txt
```

### Git Hooks & CI/CD

- **Pre-commit hooks**: Automatically run linting, formatting, and security checks
- **GitHub Actions**: Full CI/CD pipeline with quality gates
- **Conventional commits**: Enforced commit message format
- **Auto-formatting**: Code is automatically formatted on commit

### Development Workflow

1. **Make changes** to your code
2. **Stage files**: `git add .`
3. **Commit**: `git commit -m "feat: add new feature"` 
   - Pre-commit hooks automatically run
   - Code is formatted and linted
   - Security checks are performed
4. **Push**: `git push`
   - GitHub Actions run full test suite
   - Green checkmark appears when all checks pass

## Troubleshooting

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

## API Documentation

### Document Endpoints
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents  
- `DELETE /api/documents/{id}` - Delete specific document

### Chat Endpoints
- `POST /api/chat/query` - Streaming query with SSE
- `GET /api/conversations/{session_id}` - Get conversation history
- `DELETE /api/conversations/{session_id}` - Clear conversation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- GitHub Issues: [Create an issue](https://github.com/your-username/rag-demo/issues)
- Google Cloud Documentation: [Vertex AI Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- Documentation: See [Technical Guide](docs/TECHNICAL_GUIDE.md) for detailed technical information

## System Requirements

### Minimum Requirements
- **CPU**: 2 vCPUs
- **Memory**: 4GB RAM
- **Storage**: 10GB available disk space
- **Network**: Stable internet connection for API calls

### Google Cloud Requirements
- Active Google Cloud Project
- Billing enabled
- APIs enabled:
  - Vertex AI API
  - Document AI API (optional)
  - Cloud Storage API

## Security Considerations

### Authentication
- Service account credentials should have minimal required permissions
- Never commit credentials to version control
- Use environment variables or secret management systems

### Data Privacy
- Documents are processed through Google Cloud services
- Ensure compliance with your data privacy requirements
- Consider data residency requirements when selecting GCP regions

## Performance Tuning

### Backend Optimization
- **Connection Pooling**: Configure uvicorn workers based on CPU cores
- **Async Processing**: Leverage FastAPI's async capabilities
- **Caching**: Implement Redis for session management in production

### Search Optimization
- **Chunk Size**: Adjust based on document types (300-1500 chars)
- **Result Count**: Balance between coverage and performance
- **Document Filtering**: Use metadata filters to reduce search scope

### Model Selection
- **Speed Priority**: Use `gemini-2.0-flash-001`
- **Quality Priority**: Use `gemini-1.5-pro`
- **Cost Priority**: Implement caching and result reuse

## Version History

### v1.0.0 (Current)
- Initial public release
- Support for PDF, Word, Excel, CSV, and image files
- Multi-document search and chat
- Streaming responses
- Real-time document processing

### Roadmap
- [ ] Authentication and user management
- [ ] Document sharing and collaboration
- [ ] Export conversation history
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Integration with Google Drive

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`npm test` and `pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Use ESLint configuration
- **Commits**: Follow conventional commits format

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Documentation
- [Technical Guide](docs/TECHNICAL_GUIDE.md) - Detailed architecture and implementation
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

### Community
- GitHub Issues: [Create an issue](https://github.com/your-username/rag-demo/issues)
- Google Cloud Documentation: [Vertex AI Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- Stack Overflow: Tag with `vertex-ai-search` and `gemini-api`

## Acknowledgments

- Built with Google Cloud Vertex AI and Gemini
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- FastAPI framework for high-performance backend
- React and TypeScript for modern frontend development