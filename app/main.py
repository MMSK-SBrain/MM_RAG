"""
FastAPI server for the Multimodal RAG system.

Endpoints:
    GET  /health  — Liveness check + index status
    POST /ingest  — Upload and ingest a PDF into the vector store
    POST /query   — Ask a question against ingested documents
    GET  /docs    — (auto-generated) Swagger UI
"""

import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.models import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag_utils import RAGPipeline

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline when the server starts."""
    app.state.rag = RAGPipeline(settings=settings)
    logger.info("RAG pipeline initialised (mock_mode=%s)", settings.mock_mode)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal RAG API",
    description="Production-ready Multimodal Retrieval-Augmented Generation API "
    "built with Docling, LangChain, FAISS, and IBM Granite.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return system health and index status."""
    rag: RAGPipeline = app.state.rag
    return HealthResponse(
        status="healthy",
        index_loaded=rag.is_index_loaded,
        document_count=rag.document_count,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector store.

    The PDF is parsed into text, table, and image chunks,
    embedded, and added to the FAISS index.
    """
    # 1. Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: {file.filename}",
        )

    # 2. Read and validate file size
    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    # 3. Sanitise filename
    safe_filename = re.sub(r"[^\w\-.]", "_", file.filename)

    # 4. Save to temp file and ingest
    rag: RAGPipeline = app.state.rag
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", prefix=safe_filename + "_", delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        chunks_added = rag.ingest_documents([tmp_path])

        return IngestResponse(
            message="PDF ingested successfully",
            filename=safe_filename,
            chunks_added=chunks_added,
        )
    except Exception as e:
        logger.exception("Ingestion failed for %s", safe_filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Ask a question against the ingested documents.

    Uses retrieval-augmented generation: retrieves relevant chunks
    from the FAISS index and generates a grounded answer using
    IBM Granite 8B Instruct via OpenRouter.
    """
    rag: RAGPipeline = app.state.rag

    if not rag.is_index_loaded:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Use POST /ingest first.",
        )

    result = rag.query(request.question)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=request.question,
    )
