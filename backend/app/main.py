from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv
import vertexai

# Load environment variables BEFORE importing modules that use them
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now import modules that depend on environment variables
from app.api import documents, chat, websocket

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Vertex AI on startup
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    
    vertexai.init(project=project_id, location=location)
    
    logger.info(f"Vertex AI initialized for project: {project_id}")
    
    yield

app = FastAPI(
    title="NotebookLM RAG Demo",
    description="A proof-of-concept RAG application using Vertex AI",
    version="1.0.0",
    lifespan=lifespan
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
    return {"message": "NotebookLM RAG Demo API"}