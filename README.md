# RAG Engine Demo

This repository contains a Retrieval-Augmented Generation (RAG) application that pairs a FastAPI backend with a React/Vite frontend. Documents uploaded through the UI are normalized, chunked with rich metadata, indexed in Vertex AI Search, and then used to answer questions with Gemini models. The project demonstrates how to combine Google Cloud's managed retrieval stack with custom ingestion logic, document-quality safeguards, and a conversational interface.

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────┐
│ React / Vite Frontend                                  │
│ • Upload & selection UI • Streaming chat • Tailwind UI │
└───────────────▲───────────────────────┬──────────────────┘
                │                       │ SSE responses
                │ REST + uploads        │ (Gemini)
┌───────────────┴───────────────────────▼──────────────────┐
│ FastAPI Backend                                          │
│ • EnhancedDocumentProcessor      • VertexSearchService   │
│ • Optional Document AI & Gemini  • In-memory + remote    │
│   fallbacks for ingestion          chunk fallbacks       │
└───────────────▲───────────────────────┬──────────────────┘
                │                       │ Discovery Engine
                │ GCS temp storage      │ Vertex AI Search
┌───────────────┴──────────┐   ┌────────▼──────────────┐
│ Google Cloud Storage     │   │ Vertex AI Search      │
│ • Staging for uploaded   │   │ • Datastore + Serving │
│   files                   │   │   Config             │
└──────────────────────────┘   └───────────────────────┘
```

*Document ingestion* uses a hierarchical strategy: Document AI (optional) → Gemini multimodal → format-specific parsers. Chunks carry section hints, page ranges, keyword terms, and entity summaries. *Retrieval* relies on Vertex AI Search with streaming Gemini responses. When the serving index cannot satisfy a query, the backend synthesizes results from cached chunks or pulls them directly from Discovery Engine so users still receive document summaries.

## Quick Start

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 2. Create GCP resources (datastore, search engine, bucket)
./scripts/setup-gcp.sh

# 3. Copy the script output into backend/.env
#    (The script prints the exact values to use)
cp backend/.env.example backend/.env
# Then edit backend/.env with the values from step 2

# 4. Install dependencies
./scripts/setup-dev.sh

# 5. Start the backend (in one terminal)
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000

# 6. Start the frontend (in another terminal)
cd frontend && npm run dev

# 7. Open http://localhost:3000
```

## Prerequisites

- macOS, Linux, or WSL with Bash, `python3` (3.9–3.12 recommended), and `node` (16+; tested with Node 20+).
- A Google Cloud project with billing enabled.
- gcloud CLI installed and initialized (`gcloud init`).
- IAM roles: `roles/aiplatform.user`, `roles/discoveryengine.admin`, `roles/storage.objectAdmin`; add `roles/documentai.editor` if you plan to enable Document AI.

## Provisioning Google Cloud Resources

Run the bootstrap script from the repository root:

```bash
./scripts/setup-gcp.sh
```

The script will:
- Enable required APIs (Vertex AI, Discovery Engine, Document AI, Cloud Storage)
- Create a Discovery Engine datastore and search engine
- Create a GCS staging bucket
- Print environment variables to copy into `backend/.env`

**Customization:** You can customize resource names via environment variables:
```bash
export DATASTORE_ID="my-datastore"
export ENGINE_ID="my-engine"
./scripts/setup-gcp.sh
```

**If gcloud CLI lacks discovery-engine commands:** The script automatically falls back to REST API calls. No manual intervention needed.

**For detailed setup instructions** including manual Cloud Console steps, see [`docs/GCP_SETUP.md`](docs/GCP_SETUP.md).

**Document AI (optional):** Create a Form Parser processor and set `DOCAI_PROCESSOR_ID` and `DOCAI_LOCATION` in your `.env`.

## Local Environment Setup

### One-command setup (recommended)
```bash
./scripts/setup-dev.sh
```
The script creates `backend/venv`, installs editable backend requirements (including lint/test extras), installs frontend dependencies, and wires up pre-commit hooks.

### Manual setup
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd ../frontend
npm install
```

### Environment configuration
Create `backend/.env` before running the backend. See `backend/.env.example` for a complete template.

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT_ID` | ✓ | Google Cloud project that hosts Vertex AI Search and storage. |
| `GCP_LOCATION` | ✓ | **Vertex AI region** for Gemini models (e.g., `us-central1`). Do NOT use `global`. |
| `GEMINI_MODEL` | ✓ | Gemini model for generation, e.g. `gemini-2.0-flash-001`. |
| `VERTEX_SEARCH_DATASTORE_ID` | ✓ | Discovery Engine datastore ID produced by `setup-gcp.sh`. |
| `VERTEX_SEARCH_APP_ID` | ✓ | Enterprise Search app/engine ID. |
| `GCS_STAGING_BUCKET` | ✓ | GCS bucket for staging uploads (default: `${GCP_PROJECT_ID}-rag-temp`). |
| `USE_DOCUMENT_AI` | optional | `true` to run Document AI form parser before other extractors. |
| `USE_VERTEX_RANKING` | optional | Enable Vertex AI Builder re-ranking (`true`/`false`). |
| `USE_VERTEX_GROUNDING` | optional | Enable grounding calls for generated answers. |
| `DOCAI_PROCESSOR_ID` / `DOCAI_LOCATION` | optional | Required only when Document AI is enabled. |

> **Note:** Discovery Engine uses location `global` internally - this is handled by the backend code. The `GCP_LOCATION` variable is only for Vertex AI (Gemini).

## Running the Project Locally

Backend (FastAPI with Uvicorn):
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Frontend (Vite dev server on port 3000):
```bash
cd frontend
npm run dev
```

Navigate to `http://localhost:3000`, upload documents, select them, and initiate a chat session. The UI streams Gemini responses via Server-Sent Events.

## Document Processing Flow

1. **Upload** – Files are staged to `backend/uploads/` and (for larger PDFs) to `gs://{project}-rag-temp` during ingestion. Duplicate filenames are skipped gracefully.
2. **Extraction** –
   - Optional Document AI layout parsing.
   - Gemini multimodal extraction for PDFs/images within size limits.
   - File-type specific fallbacks (PyPDF2, `python-docx`, CSV, XLSX parsers).
3. **Normalization** – Text is cleaned, headings detected, tables preserved, and entities (dates, URLs, emails, currency, abbreviations) captured.
4. **Chunking** – Sentences are grouped into 300–1500 character windows with overlap and table-aware handling. Metadata for each chunk includes section hints, page ranges, keyword terms, token estimates, and table flags.
5. **Indexing** – Each chunk is indexed individually in Vertex AI Search. Metadata is also stored in memory for quick recall.
6. **Fallbacks** – If Vertex AI Search cannot satisfy a filtered query, the backend synthesizes search results from cached chunks or fetches chunk documents directly by ID so generic prompts still return context.

## Query & Chat Pipeline

- Queries with selected document IDs run through Vertex AI Search using a pre-search filter on `parent_document_id`.  
- Results are grouped by document, truncated to the optimal context size, and streamed into Gemini (`gemini-2.0-flash-001` by default).  
- Conversation history is stored per session to support follow-up questions; query reformulation injects prior context when the user references “that document” or similar pronouns.

## Key API Endpoints

| Method & Path | Purpose |
|---------------|---------|
| `POST /api/documents/upload` | Accepts multipart files, processes, and indexes them. |
| `GET /api/documents/` | Lists all files currently indexed (one row per source document). |
| `DELETE /api/documents/{id}` | Removes all chunks for a document from Vertex AI Search. |
| `POST /api/chat/query` | Streams Gemini responses (SSE) using selected document IDs. |
| `DELETE /api/conversations/{session_id}` | Clears cached conversation state. |

All endpoints respond with JSON; streaming responses are delivered as `text/event-stream`.

## Frontend Notes

The React application is written in TypeScript, styled with Tailwind, and bundled by Vite. Component entry points:
- `frontend/src/routes` – page-level routing.
- `frontend/src/components/DocumentManager.tsx` – upload flow and selection state.
- `frontend/src/components/ChatInterface.tsx` – SSE client and transcript rendering.

State is local to components; no external state libraries are required.

## Testing & Quality Gates

- **Backend** – Pytest (`pytest`), async tests via `pytest-asyncio`, static checks with Ruff, formatting with Black, import sorting via isort, and type checking with mypy.
- **Frontend** – ESLint (`npm run lint:check`), Prettier (`npm run format:check`), and TypeScript (`npm run type-check`).
- **Pre-commit** – Configured hooks run Ruff, Black, isort, mypy, ESLint, and formatting before every commit.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| "No documents found" when querying | Ensure `VERTEX_SEARCH_DATASTORE_ID` and `VERTEX_SEARCH_APP_ID` match the deployed Discovery Engine resources. Confirm the backend has network access to Google APIs. |
| Upload succeeds but chat returns an empty answer | Check backend logs for Vertex AI Search filtering messages. The fallback fetch will log cache misses and remote chunk retrieval; if both fail, verify IAM roles. |
| Slow PDF processing | Flip `USE_DOCUMENT_AI=true` for structured files or confirm file size < 19 MB for Gemini multimodal extraction. |
| Authentication errors | Re-run `gcloud auth application-default login` and confirm billing is enabled. |
| Frontend cannot connect to backend | Ensure the backend is running on port 8000 and CORS is enabled (FastAPI config allows the local dev origin by default). |
| "Unsupported region for Vertex AI" error | Set `GCP_LOCATION=us-central1` (or another valid region) in `.env`. Do NOT use `global`. |
| `gcloud discovery-engine` not found | The setup script automatically uses REST API fallback. No action needed. See `docs/GCP_SETUP.md` for manual options. |
| Cloud Console asks for bucket when creating datastore | Select "No data source" or create an empty bucket. The app imports documents via API, not bucket sync. Sync frequency is irrelevant. |

## Repository Layout

```
.
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI routers for documents & chat
│   │   ├── services/            # Ingestion, Vertex AI Search, Gemini helpers
│   │   └── utils/
│   ├── requirements.txt         # Runtime dependencies
│   └── pyproject.toml           # Packaging + tooling configuration
├── frontend/
│   ├── src/                     # React components and routes
│   ├── public/
│   └── vite.config.ts
├── scripts/                     # setup-dev.sh, setup-gcp.sh helpers
└── docs/                        # Reference material (architecture, etc.)
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
