# CLAUDE.MD - NotebookLM-style RAG Demo Build Prompt

## Project Context

I need to build a NotebookLM-style RAG (Retrieval Augmented Generation) demo using Google Cloud Platform and Gemini. This is a proof-of-concept for a customer who wants to replicate Google's NotebookLM experience for document Q&A.

## Key Considerations
- never hallucinate. If you do not know how to use an API, or you're unsure of the methods or classes. Please ask me what you need and I will provide this information for you. This is **vital**. 

### Key Requirements
- **Scale**: Small proof of concept (<100 documents)
- **Document Types**: PDFs, images, CSV files
- **UI**: Web interface with slide-out chat (similar to Gemini integration in Google products)
- **Performance**: ~10 seconds processing time is acceptable
- **Features**: Upload documents, select documents from page, ask questions about content

### Architecture Decision: Vertex AI RAG Engine
Based on latest research, we'll use **Vertex AI RAG Engine** which is a fully managed service that handles:
- Vector storage (no external database needed)
- Document chunking (automatic)
- Embedding generation (automatic)
- Retrieval and ranking (automatic)

This is now GA (Generally Available) and provides managed orchestration for RAG applications.

**Important**: For <100 documents, you might not need RAG at all. Gemini 2.0 Flash can handle ~1,500 pages in a single context window. Consider:
- Use RAG Engine when you need to search across many documents
- Use direct context when processing individual documents or small sets

## Quick Start Flow

1. **Enable GCP APIs** → `gcloud services enable aiplatform.googleapis.com`
2. **Create RAG Corpus** → One-time setup in Vertex AI
3. **Upload Documents** → RAG Engine handles chunking/embedding automatically
4. **Query Documents** → RAG Engine retrieves relevant chunks automatically
5. **Generate Response** → Gemini uses retrieved context to answer

No databases, no manual embeddings, no complex setup!

## Build Instructions for Claude Code

When I ask you to build this RAG demo, please create a complete implementation with the following components:

### 1. Backend API (Python/FastAPI)
Create a FastAPI backend that includes:
- Document upload endpoints supporting PDF, images, and CSV
- Document processing pipeline using Gemini 2.0 Flash
- Vector storage using Cloud SQL with pgvector
- Query endpoint with semantic search and RAG
- Caching layer using Redis for performance
- WebSocket support for real-time chat

### 2. Frontend (React/TypeScript)
Build a React application with:
- Document list panel on the left (25% width)
- Main document viewer in center
- Slide-out chat drawer on right (triggered by button)
- Drag-and-drop file upload
- PDF preview using react-pdf
- Real-time chat interface using @chatscope/chat-ui-kit-react
- Material-UI v6 for styling

### 3. Simple Setup for POC
- Create a GCP project and enable Vertex AI API
- Set environment variables:
  ```bash
  export GCP_PROJECT_ID="your-project-id"
  export GCP_LOCATION="us-central1"
  ```
- Create service account with Vertex AI permissions
- One-time RAG corpus creation:
  ```python
  rag_corpus = rag.create_corpus(display_name="notebooklm_demo")
  ```
That's it! No database setup, no infrastructure complexity.

### 4. Key Technical Decisions
Use these specific choices:
- **LLM**: Gemini 2.0 Flash (`gemini-2.0-flash-exp`) 
- **RAG Service**: Vertex AI RAG Engine (fully managed, no external databases)
- **Alternative**: For <100 docs, consider direct context processing (no RAG needed)
- **Frontend Framework**: React 19 with TypeScript
- **UI Components**: shadcn/ui + Tailwind CSS
- **Icons**: Lucide React
- **No Additional Infrastructure**: No databases, no caching, no complex setup

### 5. Code Structure
```
project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── services/
│   │   │   ├── rag_engine.py      # Vertex AI RAG Engine integration
│   │   │   ├── document_uploader.py
│   │   │   └── gemini_client.py
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── websocket.py
│   │   └── utils/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/               # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── sheet.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   └── ...
│   │   │   ├── DocumentList.tsx
│   │   │   ├── DocumentViewer.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   └── FileUploader.tsx
│   │   ├── lib/
│   │   │   └── utils.ts         # cn() utility
│   │   ├── hooks/
│   │   └── App.tsx
│   ├── components.json          # shadcn/ui config
│   └── package.json
└── README.md
```

### 6. Core Implementation Patterns

#### RAG Engine Setup
```python
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

# Initialize once
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create corpus (one-time setup)
rag_corpus = rag.create_corpus(display_name="notebooklm_demo")

# Upload documents - RAG Engine handles everything
async def upload_document(file_path: str, display_name: str):
    rag.upload_file(
        corpus_name=rag_corpus.name,
        path=file_path,
        display_name=display_name
    )
```

#### Query Implementation
```python
async def query_documents(question: str, corpus_name: str):
    # Create retrieval tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_corpora=[corpus_name],
                similarity_top_k=10,
                vector_distance_threshold=0.5
            )
        )
    )
    
    # Query with automatic retrieval
    rag_model = GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=[rag_retrieval_tool]
    )
    
    # Stream response
    response = rag_model.generate_content_stream(question)
    return response
```

#### Alternative: Direct Context Processing
For small document sets, implement this simpler approach:
```python
async def query_direct(document_content: str, question: str):
    model = GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        document_content,  # Up to 1M tokens
        f"Answer based on the document: {question}"
    ])
    return response
```

#### Main App Layout
```tsx
import { cn } from "@/lib/utils"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"

function App() {
  return (
    <div className="flex h-screen bg-background">
      {/* Document List - 25% width */}
      <div className="w-1/4 min-w-[300px] border-r">
        <div className="p-4">
          <h2 className="text-lg font-semibold">Documents</h2>
        </div>
        <Separator />
        <ScrollArea className="h-[calc(100vh-60px)]">
          <DocumentList />
        </ScrollArea>
      </div>
      
      {/* Document Viewer - Remaining space */}
      <div className="flex-1 flex flex-col">
        <DocumentViewer />
      </div>
      
      {/* Chat Interface - Slide-out sheet */}
      <ChatInterface />
    </div>
  )
}
```

#### File Upload Component
```tsx
import { useDropzone } from 'react-dropzone'
import { Card } from "@/components/ui/card"
import { Upload, FileText, Image, Table } from "lucide-react"
import { cn } from "@/lib/utils"

function FileUploader() {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg'],
      'text/csv': ['.csv']
    },
    onDrop: async (files) => {
      // Upload logic
    }
  })

  return (
    <Card 
      {...getRootProps()} 
      className={cn(
        "border-2 border-dashed p-8 text-center cursor-pointer transition-colors",
        isDragActive && "border-primary bg-primary/5"
      )}
    >
      <input {...getInputProps()} />
      <Upload className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">
        Drop files here or click to browse
      </p>
      <div className="flex gap-2 justify-center mt-2">
        <FileText className="h-4 w-4" />
        <Image className="h-4 w-4" />
        <Table className="h-4 w-4" />
      </div>
    </Card>
  )
}
```

#### Handling Dynamic PDF Structures
The Vertex AI RAG Engine automatically handles different PDF structures, but you can add metadata for better retrieval:
```python
# Add document-specific metadata
rag.upload_file(
    corpus_name=rag_corpus.name,
    path="financial_report.pdf",
    rag_file=rag.RagFile(
        display_name="Q4 2024 Financial Report",
        description="Contains tables, charts, and narrative sections"
    )
)
```

#### Frontend Chat Component
```tsx
import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MessageSquare } from "lucide-react"

function ChatInterface() {
  const [open, setOpen] = useState(false)
  
  return (
    <>
      <Button 
        onClick={() => setOpen(true)}
        className="fixed top-4 right-4"
      >
        <MessageSquare className="h-4 w-4 mr-2" />
        Chat
      </Button>
      
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent className="w-[400px] sm:w-[540px]">
          <SheetHeader>
            <SheetTitle>Document Q&A</SheetTitle>
          </SheetHeader>
          <ScrollArea className="h-[calc(100vh-200px)] mt-6">
            {/* Chat messages */}
          </ScrollArea>
          <div className="absolute bottom-4 left-4 right-4">
            <Input placeholder="Ask about your documents..." />
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
```

- Implement streaming responses with skeleton loaders
- Show which documents were used for the answer
- Support follow-up questions with context
- Use Lucide icons for visual feedback

### 7. Performance Optimizations
- Use RAG Engine's built-in optimizations
- Implement response streaming for better UX
- For documents <10 pages, use direct context instead of RAG
- Use React.memo and useMemo for UI performance
- Virtualize long document lists with @tanstack/react-virtual
- Use shadcn/ui Skeleton components for loading states

### 8. GCP APIs to Enable
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
```

## Example Prompts for Testing

Once built, test with these scenarios:
1. "What are the key findings in this document?"
2. "Compare the data between these two CSV files"
3. "Summarize all action items from the uploaded PDFs"
4. "What does the chart on page 5 show?"
5. "Extract all financial metrics from the quarterly report"

## Additional Context

- Latest versions as of June 2025:
  - google-cloud-aiplatform>=1.71.1
  - vertexai>=1.71.1
  - react: ^19.0.0
  - typescript: ^5.7.2
  - tailwindcss: ^3.4.0
  - shadcn/ui components (installed via CLI)

- Key Documentation:
  - [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview)
  - [Gemini Models](https://cloud.google.com/vertex-ai/generative-ai/docs/models)
  - [shadcn/ui](https://ui.shadcn.com/)

## Build Command

When ready to build, I'll say: "Build the complete NotebookLM RAG demo using Vertex AI RAG Engine following the CLAUDE.MD specifications"