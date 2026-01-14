import logging
import os
from contextlib import asynccontextmanager

import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables BEFORE importing modules that use them
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI BEFORE importing modules that use GenerativeModel
# This is required because some modules create GenerativeModel instances at import time
project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "us-central1")

if not project_id:
    raise ValueError("GCP_PROJECT_ID environment variable is required")

vertexai.init(project=project_id, location=location)
logger.info(f"Vertex AI initialized for project: {project_id}, location: {location}")

# Now import modules that depend on Vertex AI being initialized
from app.api import chat, documents, websocket


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Vertex AI already initialized above
    yield


app = FastAPI(
    title="RAG Demo API",
    description="A production-ready RAG application using Vertex AI Search and Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.get("/")
async def root():
    return {"message": "RAG Demo API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-demo-backend"}
