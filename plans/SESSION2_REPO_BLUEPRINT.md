# Session 2 Repo Blueprint — Production-Ready Multimodal RAG

## Purpose of This Document

This is a **build specification** for Claude Code (or any developer) to construct a complete, working FastAPI-based Multimodal RAG repository. The repo serves two purposes:

1. **Instructor demo base** — every Session 2 topic (CI/CD, security, evaluation) is layered onto this repo live during the bootcamp.
2. **Reference implementation** — shows learners what a well-structured assignment submission looks like.

---

## 1. Source Repo & What to Preserve

### Original Repo

**URL:** `https://github.com/soham0701/Multimodal_RAG`

**Original file structure:**

```
Multimodal_RAG/
├── rag_utils.py            # Core RAG logic
├── app.py                  # Gradio UI (will be REPLACED by FastAPI)
├── initial_ingest.py       # CLI bulk ingestion script
├── source_documents/       # 8 sample multimodal PDFs
├── requirements.txt
├── .env.example
└── .devcontainer/
    └── devcontainer.json   # GitHub Codespaces config
```

### What the original `rag_utils.py` does (reconstruct from this spec)

The file contains the core RAG pipeline with these components:

**1. Document Parsing (using Docling):**
- Uses `DocumentConverter` from `docling` to parse PDFs
- Extracts three chunk types:
  - **Text chunks**: Uses `HybridChunker` from `docling.chunking` to split text into semantically meaningful chunks
  - **Table chunks**: Extracted as Markdown representations
  - **Image chunks**: Images are extracted, then summarised using IBM Granite Vision 2B model via Replicate API. The summary text becomes the chunk content.
- Each chunk is stored as a LangChain `Document` object with metadata (source file, chunk type, page number)

**2. Embedding & Indexing:**
- Uses IBM Granite Embedding 30M model (`ibm-granite/granite-embedding-30m-english`) via HuggingFace `sentence-transformers`
- Embeds all chunks (text, table markdown, image summaries) into a FAISS vector index
- FAISS index is persisted to disk (`faiss_index/` directory)

**3. Query Pipeline:**
- Uses LangChain `RetrievalQA` chain
- Retriever: FAISS similarity search (top-k=4)
- LLM: IBM Granite 8B Instruct via Replicate API
- Custom grounding prompt template that instructs the model to only answer from retrieved context
- The prompt template is defined as a module-level constant

**4. Key implementation details from Session 1 optimisations:**
- Vision processor is hoisted to module level (not re-created per image)
- Uses `.invoke()` instead of deprecated `.run()`
- All dead code removed, docstrings added

### What to KEEP from the original

- The entire RAG pipeline logic (parsing, embedding, indexing, querying)
- The `source_documents/` folder with sample PDFs
- The `.devcontainer/devcontainer.json` for Codespaces
- The `.env.example` pattern

### What to REPLACE

- `app.py` (Gradio) → `app/main.py` (FastAPI)
- `initial_ingest.py` (CLI) → integrated into FastAPI `/ingest` endpoint

---

## 2. Target Repo Structure

After the full build is complete (all layers applied), the repo should look like this:

```
multimodal-rag-production/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application with all 4 endpoints
│   ├── rag_utils.py               # Core RAG logic (adapted from original)
│   ├── config.py                  # Pydantic Settings — loads .env
│   └── models.py                  # Pydantic request/response schemas
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                # pytest tests for FastAPI endpoints (TestClient)
│   ├── test_hammering.py          # Rate limit stress test script
│   └── conftest.py                # Shared fixtures (test client, mock data)
│
├── evaluation/
│   ├── demo_evaluation.py         # RAGAS evaluation script
│   └── sample_eval_data.json      # Sample questions + ground truth for eval demo
│
├── source_documents/              # Sample multimodal PDFs (from original repo)
│   └── (8 PDF files)
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
│
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces configuration
│
├── faiss_index/                   # Created at runtime by /ingest (gitignored)
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Dev/test dependencies (ruff, pytest, ragas)
└── README.md
```

---

## 3. Build Order (Matches Demo Flow)

The instructor's live demo order is:

```
Phase 1: Base App        →  Pull repo, get it running, test manually
Phase 2: CI/CD           →  Automate linting + testing + Docker build
Phase 3: Security        →  Add rate limiting, input validation, key management
Phase 4: Evaluation      →  Add RAGAS evaluation harness
```

### Build the repo in this exact order:

### Phase 1 — Base FastAPI App

Build these files first. The app must be fully functional before adding any Session 2 layers.

**Files to create:**
- `app/__init__.py`
- `app/config.py`
- `app/models.py`
- `app/rag_utils.py`
- `app/main.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`

Copy from original repo:
- `source_documents/` (all 8 PDFs)
- `.devcontainer/devcontainer.json`

### Phase 2 — CI/CD Pipeline + Tests

**Files to create:**
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_api.py`
- `requirements-dev.txt`
- `.github/workflows/ci.yml`

### Phase 3 — Security Hardening

**Files to modify:**
- `app/main.py` — add rate limiting middleware (slowapi), tighten input validation
- `app/models.py` — add stricter validators

**Files to create:**
- `tests/test_hammering.py`

### Phase 4 — Evaluation

**Files to create:**
- `evaluation/demo_evaluation.py`
- `evaluation/sample_eval_data.json`

### Phase 5 — Containerisation

**Files to create:**
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

---

## 4. Detailed File Specifications

### 4.1 `app/config.py`

Uses `pydantic-settings` to load configuration from environment variables.

```python
"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """All configuration is loaded from .env or environment variables."""

    # --- LLM / Replicate ---
    replicate_api_token: str = ""

    # --- Embedding model ---
    embedding_model_name: str = "ibm-granite/granite-embedding-30m-english"

    # --- Granite models on Replicate ---
    granite_instruct_model: str = "ibm-granite/granite-3.1-8b-instruct"
    granite_vision_model: str = "ibm-granite/granite-vision-3.1-2b"

    # --- FAISS ---
    faiss_index_path: str = "faiss_index"

    # --- RAG parameters ---
    retriever_top_k: int = 4
    max_query_length: int = 1000

    # --- Rate limiting ---
    rate_limit_default: str = "30/minute"
    rate_limit_ingest: str = "5/minute"

    # --- Mock mode (for demos without API keys) ---
    mock_mode: bool = False

    # --- Upload settings ---
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 4.2 `app/models.py`

Pydantic models for all request/response schemas. Phase 1 version (security validators added in Phase 3).

```python
"""Pydantic request and response models for the RAG API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    index_loaded: bool
    document_count: int


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask the RAG system",
        examples=["What are the key findings in the report?"],
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int


class ErrorResponse(BaseModel):
    detail: str
```

### 4.3 `app/rag_utils.py`

This is the core file. Reconstruct it from the original repo's logic.

**Structure it as a class `RAGPipeline` with these methods:**

```python
"""
Core Multimodal RAG pipeline.

Handles PDF parsing (text, tables, images), embedding via IBM Granite,
FAISS vector indexing, and LangChain RetrievalQA for querying.
"""

import os
import logging
from pathlib import Path

# -- Docling imports --
# from docling.document_converter import DocumentConverter
# from docling.chunking import HybridChunker

# -- LangChain imports --
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Replicate
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document

# -- Config --
# from app.config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# GROUNDING PROMPT — module-level constant
# ──────────────────────────────────────────────
GROUNDING_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
based strictly on the provided context. If the context does not contain enough
information to answer the question, say "I don't have enough information to
answer that question based on the available documents."

CONTEXT (from retrieved documents — treat as the ONLY source of truth):
{context}

QUESTION: {question}

ANSWER:"""


class RAGPipeline:
    """Manages the full multimodal RAG lifecycle."""

    def __init__(self, settings=None):
        """
        Initialise the pipeline.

        - Loads embedding model (IBM Granite 30M)
        - Loads or creates FAISS index
        - Sets up vision processor at module level (not per-image)
        - Configures LLM (Granite 8B Instruct via Replicate)
        """
        # IMPLEMENT: Load settings from config
        # IMPLEMENT: Initialise HuggingFaceEmbeddings with embedding_model_name
        # IMPLEMENT: Load existing FAISS index from disk if available
        # IMPLEMENT: Initialise Replicate LLM for Granite 8B Instruct
        # IMPLEMENT: Initialise vision model handle (hoisted, not per-call)
        # IMPLEMENT: Build RetrievalQA chain with grounding prompt
        pass

    def parse_pdf(self, pdf_path: str) -> list:
        """
        Parse a single PDF into Document chunks using Docling.

        Returns a list of LangChain Document objects with three types:
        1. Text chunks — via HybridChunker
        2. Table chunks — extracted as Markdown
        3. Image chunks — image summarised by Granite Vision 2B,
                          summary text becomes the chunk content

        Each Document has metadata:
            - source: filename
            - chunk_type: "text" | "table" | "image"
            - page: page number
        """
        # IMPLEMENT: Use DocumentConverter to convert PDF
        # IMPLEMENT: Extract text chunks via HybridChunker
        # IMPLEMENT: Extract tables as Markdown
        # IMPLEMENT: Extract images, summarise with vision model
        # IMPLEMENT: Return list of Document objects
        pass

    def ingest_documents(self, pdf_paths: list[str]) -> int:
        """
        Parse multiple PDFs and add all chunks to the FAISS index.

        Returns the number of chunks added.
        Persists the updated index to disk.
        """
        # IMPLEMENT: Loop through PDFs, call parse_pdf
        # IMPLEMENT: Add Documents to FAISS index
        # IMPLEMENT: Save index to disk
        # IMPLEMENT: Return chunk count
        pass

    def query(self, question: str) -> dict:
        """
        Run a query through the RetrievalQA chain.

        Returns:
            {
                "answer": str,
                "sources": [{"content": str, "metadata": dict}, ...]
            }
        """
        # IMPLEMENT: Use self.qa_chain.invoke({"query": question})
        # IMPLEMENT: Extract source documents from result
        # IMPLEMENT: Return structured response
        pass

    @property
    def document_count(self) -> int:
        """Return the number of documents/chunks in the FAISS index."""
        # IMPLEMENT: Return index size or 0 if no index
        pass

    @property
    def is_index_loaded(self) -> bool:
        """Return True if a FAISS index is loaded in memory."""
        pass
```

**CRITICAL — Mock Mode:**

The class must support `MOCK_MODE=true` from settings. When mock mode is enabled:

- `__init__` does NOT load any models or FAISS index
- `parse_pdf` returns 3 fake Document objects (one text, one table, one image chunk)
- `ingest_documents` increments an in-memory counter and returns a fake chunk count
- `query` returns a canned response like:

```json
{
    "answer": "Based on the retrieved documents, the key finding is that multimodal RAG systems improve accuracy by 35% when combining text, table, and image data. [MOCK RESPONSE — enable real mode by setting MOCK_MODE=false]",
    "sources": [
        {"content": "Mock text chunk content...", "metadata": {"source": "sample.pdf", "chunk_type": "text", "page": 1}},
        {"content": "Mock table chunk content...", "metadata": {"source": "sample.pdf", "chunk_type": "table", "page": 3}}
    ]
}
```

This allows ALL demos (rate limiting, testing, CI/CD, Docker) to run without Replicate API keys.

### 4.4 `app/main.py`

**Phase 1 version** — clean FastAPI app with the 4 required endpoints. Security middleware is added in Phase 3.

```python
"""
FastAPI server for the Multimodal RAG system.

Endpoints:
    GET  /health  — Liveness check + index status
    POST /ingest  — Upload and ingest a PDF into the vector store
    POST /query   — Ask a question against ingested documents
    GET  /docs    — (auto-generated) Swagger UI
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.config import get_settings
from app.models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    IngestResponse,
)
from app.rag_utils import RAGPipeline

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan: initialise RAG pipeline once at startup ──
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


# ── GET /health ──
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return system health and index status."""
    rag: RAGPipeline = app.state.rag
    return HealthResponse(
        status="healthy",
        index_loaded=rag.is_index_loaded,
        document_count=rag.document_count,
    )


# ── POST /ingest ──
@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector store.

    The PDF is parsed into text, table, and image chunks,
    embedded, and added to the FAISS index.
    """
    # IMPLEMENT:
    # 1. Validate file extension (.pdf only)
    # 2. Validate file size (< max_upload_size_mb)
    # 3. Save to temp location
    # 4. Call rag.ingest_documents([temp_path])
    # 5. Clean up temp file
    # 6. Return IngestResponse with chunk count
    pass


# ── POST /query ──
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Ask a question against the ingested documents.

    Uses retrieval-augmented generation: retrieves relevant chunks
    from the FAISS index and generates a grounded answer using
    IBM Granite 8B Instruct.
    """
    rag: RAGPipeline = app.state.rag

    if not rag.is_index_loaded:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Use POST /ingest first.",
        )

    # IMPLEMENT:
    # 1. Call rag.query(request.question)
    # 2. Return QueryResponse
    pass
```

**Important implementation notes for main.py:**
- Use `asynccontextmanager` lifespan (not deprecated `@app.on_event`)
- File upload in `/ingest` should use `UploadFile` with temp file pattern:
  - Write uploaded bytes to a `tempfile.NamedTemporaryFile(suffix=".pdf")`
  - Pass temp path to `rag.ingest_documents()`
  - Clean up with `os.unlink()` in a `finally` block
- The `/docs` endpoint is automatic (FastAPI's built-in Swagger UI at `/docs`)

### 4.5 `requirements.txt` (Production)

```
# -- Web framework --
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9

# -- RAG core --
docling>=2.5.0
langchain>=0.3.0
langchain-community>=0.3.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
replicate>=0.34.0

# -- Config --
pydantic-settings>=2.5.0
python-dotenv>=1.0.1

# -- Security (added in Phase 3, but include from start) --
slowapi>=0.1.9
```

### 4.6 `requirements-dev.txt`

```
-r requirements.txt

# -- Testing --
pytest>=8.3.0
httpx>=0.27.0

# -- Linting --
ruff>=0.6.0

# -- Evaluation (Phase 4) --
ragas>=0.2.0
datasets>=3.0.0
```

### 4.7 `.env.example`

```bash
# ── LLM API ──
REPLICATE_API_TOKEN=r8_your_token_here

# ── Mock mode (set to true for demos without API keys) ──
MOCK_MODE=true

# ── Model configuration (defaults are fine for most cases) ──
# EMBEDDING_MODEL_NAME=ibm-granite/granite-embedding-30m-english
# GRANITE_INSTRUCT_MODEL=ibm-granite/granite-3.1-8b-instruct
# GRANITE_VISION_MODEL=ibm-granite/granite-vision-3.1-2b
# FAISS_INDEX_PATH=faiss_index
# RETRIEVER_TOP_K=4

# ── Security ──
# RATE_LIMIT_DEFAULT=30/minute
# RATE_LIMIT_INGEST=5/minute
# MAX_UPLOAD_SIZE_MB=50
```

### 4.8 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# Environment
.env

# FAISS index (generated at runtime)
faiss_index/

# Temp uploads
temp_uploads/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Docker
*.log
```

### 4.9 `.devcontainer/devcontainer.json`

Adapt from the original repo. Ensure it works for the new FastAPI structure:

```json
{
    "name": "Multimodal RAG - Production",
    "image": "mcr.microsoft.com/devcontainers/python:3.10",
    "features": {
        "ghcr.io/devcontainers/features/docker-in-docker:2": {}
    },
    "postCreateCommand": "pip install -r requirements.txt && pip install -r requirements-dev.txt",
    "forwardPorts": [8000],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "charliermarsh.ruff",
                "ms-azuretools.vscode-docker"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python",
                "python.testing.pytestEnabled": true,
                "python.testing.pytestArgs": ["tests/"]
            }
        }
    }
}
```

**NOTE:** The `docker-in-docker` feature is critical — it allows running Docker commands inside Codespaces for the containerisation demo.

---

## 5. Phase 2 — CI/CD & Tests (Detailed Specs)

### 5.1 `tests/conftest.py`

```python
"""Shared pytest fixtures."""

import os
import pytest
from fastapi.testclient import TestClient

# Force mock mode for all tests
os.environ["MOCK_MODE"] = "true"

from app.main import app  # noqa: E402 — must import after env var is set


@pytest.fixture
def client():
    """FastAPI test client with mock RAG pipeline."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_pdf_path():
    """Path to a real sample PDF from source_documents/."""
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "source_documents")
    pdfs = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    if pdfs:
        return os.path.join(docs_dir, pdfs[0])
    return None
```

### 5.2 `tests/test_api.py`

Should contain **6–8 test functions** covering:

| Test | What It Validates |
|------|-------------------|
| `test_health_endpoint` | GET /health returns 200 with correct schema |
| `test_health_response_schema` | Response has `status`, `index_loaded`, `document_count` |
| `test_query_with_valid_question` | POST /query with valid question returns 200 |
| `test_query_empty_string` | POST /query with empty `question` returns 422 |
| `test_query_too_long` | POST /query with >1000 char question returns 422 |
| `test_query_missing_field` | POST /query with no `question` field returns 422 |
| `test_ingest_non_pdf` | POST /ingest with a .txt file returns 400 |
| `test_ingest_valid_pdf` | POST /ingest with a real PDF returns 200 with chunk count |
| `test_docs_endpoint` | GET /docs returns 200 (Swagger UI) |

**Implementation notes:**
- Use `TestClient` from `fastapi.testclient` (which uses `httpx` under the hood)
- For file upload tests, use `client.post("/ingest", files={"file": ("test.pdf", open(path, "rb"), "application/pdf")})`
- For the non-PDF test, create a tiny in-memory bytes object: `("test.txt", b"hello", "text/plain")`
- All tests run against mock mode — no real LLM calls

### 5.3 `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with ruff
        run: ruff check .

      - name: Run tests
        env:
          MOCK_MODE: "true"
        run: pytest tests/ -v --tb=short

  docker-build:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t multimodal-rag-app .

      - name: Verify container starts
        run: |
          docker run -d --name test-container \
            -e MOCK_MODE=true \
            -p 8000:8000 \
            multimodal-rag-app
          sleep 5
          curl -f http://localhost:8000/health || exit 1
          docker stop test-container
```

**Key design decisions:**
- Two jobs: `lint-and-test` runs first; `docker-build` only runs if tests pass (`needs:`)
- Mock mode is enforced in CI via environment variable
- Docker job does a **smoke test** — starts the container and hits `/health`
- No secrets needed in CI since everything runs in mock mode

---

## 6. Phase 3 — Security Hardening (Detailed Specs)

### 6.1 Modifications to `app/main.py`

Add these on top of the Phase 1 version:

**Rate limiting with slowapi:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Then decorate endpoints:
@app.post("/query", ...)
@limiter.limit(settings.rate_limit_default)    # "30/minute"
async def query_rag(request: Request, body: QueryRequest):
    ...

@app.post("/ingest", ...)
@limiter.limit(settings.rate_limit_ingest)     # "5/minute"
async def ingest_pdf(request: Request, file: UploadFile = File(...)):
    ...
```

**Note:** slowapi requires `Request` as the first parameter in decorated functions. Update the function signatures accordingly.

**Input validation hardening in `/ingest`:**

```python
async def ingest_pdf(request: Request, file: UploadFile = File(...)):
    # 1. Check file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: {file.filename}",
        )

    # 2. Check file size (read in chunks to avoid memory issues)
    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    # 3. Sanitise filename (prevent path traversal)
    import re
    safe_filename = re.sub(r'[^\w\-.]', '_', file.filename)
    ...
```

### 6.2 `tests/test_hammering.py`

This is NOT a pytest file — it's a **standalone script** for the live demo. It sends rapid requests to demonstrate rate limiting.

```python
"""
Rate limit demonstration script.

Run against a live server to show rate limiting in action.
Usage: python tests/test_hammering.py
"""

import requests
import time

BASE_URL = "http://localhost:8000"
ENDPOINT = "/query"
PAYLOAD = {"question": "What is the summary of the document?"}
NUM_REQUESTS = 50
DELAY = 0.1  # 100ms between requests

def main():
    print(f"Sending {NUM_REQUESTS} requests to {BASE_URL}{ENDPOINT}...")
    print(f"Delay between requests: {DELAY}s\n")

    results = {"success": 0, "rate_limited": 0, "error": 0}

    for i in range(1, NUM_REQUESTS + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}{ENDPOINT}",
                json=PAYLOAD,
                timeout=5,
            )
            if resp.status_code == 200:
                results["success"] += 1
                status = "✓ 200 OK"
            elif resp.status_code == 429:
                results["rate_limited"] += 1
                status = "✗ 429 TOO MANY REQUESTS"
            else:
                results["error"] += 1
                status = f"? {resp.status_code}"
            print(f"  Request {i:3d}/{NUM_REQUESTS}: {status}")
        except requests.exceptions.RequestException as e:
            results["error"] += 1
            print(f"  Request {i:3d}/{NUM_REQUESTS}: ERROR — {e}")
        time.sleep(DELAY)

    print(f"\n{'='*40}")
    print(f"Results:")
    print(f"  Successful:    {results['success']}")
    print(f"  Rate limited:  {results['rate_limited']}")
    print(f"  Errors:        {results['error']}")

if __name__ == "__main__":
    main()
```

---

## 7. Phase 4 — Evaluation (Detailed Specs)

### 7.1 `evaluation/sample_eval_data.json`

A JSON file with 5 sample evaluation entries. Each entry has:

```json
[
    {
        "question": "What are the main findings discussed in the document?",
        "ground_truth": "The document discusses improvements in multimodal processing leading to a 35% accuracy increase.",
        "contexts": ["Text chunk about findings...", "Table chunk with metrics..."],
        "answer": "The main findings show a 35% improvement in accuracy through multimodal processing."
    },
    {
        "question": "What data is shown in Table 2?",
        "ground_truth": "Table 2 shows performance metrics across three model variants.",
        "contexts": ["| Model | Accuracy | F1 |\n|---|---|---|\n| A | 0.85 | 0.82 |"],
        "answer": "Table 2 presents performance metrics including accuracy and F1 scores for three model variants."
    }
]
```

Include 5 entries — a mix of text-based, table-based, and image-based questions. Make them generic enough to work as examples but specific enough to be realistic.

### 7.2 `evaluation/demo_evaluation.py`

A standalone script (~40–50 lines) that:

1. Loads eval data from `sample_eval_data.json`
2. Creates a RAGAS `Dataset` from the entries
3. Runs RAGAS evaluation with 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall
4. Prints a formatted scorecard to the terminal
5. Optionally runs against the live API (if `--live` flag is passed) by sending real queries to `/query`

```python
"""
RAGAS Evaluation Demo for Multimodal RAG.

Usage:
    python evaluation/demo_evaluation.py                  # Use sample data
    python evaluation/demo_evaluation.py --live            # Query live API
"""

# IMPLEMENT:
# 1. Load sample_eval_data.json
# 2. Convert to RAGAS Dataset format
# 3. Run evaluate() with metrics:
#    - faithfulness
#    - answer_relevancy
#    - context_precision
#    - context_recall
# 4. Print scorecard
#
# If --live flag:
#    - For each question in eval data, POST to http://localhost:8000/query
#    - Use the actual API response as the "answer"
#    - Use retrieved sources as "contexts"
#    - Still use ground_truth from the JSON file
#    - Run RAGAS on the live results

# Print format:
# ┌─────────────────────┬───────┐
# │ Metric              │ Score │
# ├─────────────────────┼───────┤
# │ Faithfulness        │  0.85 │
# │ Answer Relevancy    │  0.79 │
# │ Context Precision   │  0.90 │
# │ Context Recall      │  0.72 │
# └─────────────────────┴───────┘
```

**Note on RAGAS + evaluator LLM:** RAGAS needs an LLM to compute its metrics (faithfulness, relevancy etc. are LLM-judged). By default RAGAS uses OpenAI. For this demo, configure RAGAS to use the same Replicate/Granite model or provide instructions to set `OPENAI_API_KEY` as an alternative evaluator. Document this clearly in the script's docstring.

---

## 8. Phase 5 — Containerisation (Detailed Specs)

### 8.1 `Dockerfile`

```dockerfile
# ── Base image ──
FROM python:3.10-slim

# ── System dependencies ──
# Some ML libraries (FAISS, sentence-transformers) need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached layer) ──
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──
COPY . .

# ── Runtime ──
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Instructor talking points to embed in handbook:**
- `python:3.10-slim` — not Alpine, because NumPy/FAISS wheels don't compile on Alpine
- `COPY requirements.txt .` before `COPY . .` — Docker layer caching optimisation
- `--no-cache-dir` — keeps image ~200MB smaller
- `0.0.0.0` — mandatory inside containers; `localhost` won't accept external connections
- `EXPOSE 8000` — documentation only; does not publish the port

### 8.2 `docker-compose.yml`

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./source_documents:/app/source_documents:ro
      - faiss_data:/app/faiss_index
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  faiss_data:
```

**Notes:**
- `source_documents` mounted read-only (`:ro`) — the app reads PDFs but shouldn't modify them
- `faiss_data` is a named volume — persists index across container restarts
- Healthcheck uses the `/health` endpoint we built
- No separate vector DB service needed since we use FAISS (file-based). The docker-compose.yml in the planning doc showed ChromaDB — that's a "stretch" version for the talking points only. Keep the default simple with just the app service.

### 8.3 `.dockerignore`

```
.git
.github
__pycache__
*.pyc
.env
faiss_index/
.venv
venv
tests/
evaluation/
.devcontainer
*.md
```

---

## 9. Environment Setup Instructions

### 9.1 Local Development (Your Dev Machine)

```bash
# 1. Clone the original repo
git clone https://github.com/soham0701/Multimodal_RAG.git
cd Multimodal_RAG

# 2. You will restructure this into the new layout using Claude Code
#    (or create a fresh repo and copy source_documents/ from the original)

# 3. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Set up environment
cp .env.example .env
# Edit .env — set MOCK_MODE=true for initial testing

# 6. Run the server
uvicorn app.main:app --reload --port 8000

# 7. Test it
curl http://localhost:8000/health
curl http://localhost:8000/docs    # Opens Swagger UI

# 8. Run tests
pytest tests/ -v

# 9. Run linting
ruff check .
```

### 9.2 GitHub Codespaces (Mock Production)

```bash
# 1. Push the completed repo to GitHub

# 2. Open in Codespaces:
#    GitHub repo page → Code → Codespaces → Create codespace on main

# 3. Codespaces auto-runs postCreateCommand from devcontainer.json
#    (installs all dependencies)

# 4. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Codespaces auto-forwards port 8000 — click the URL in the terminal

# 6. For Docker demos inside Codespaces:
docker build -t my-rag-app .
docker run -p 8001:8000 -e MOCK_MODE=true my-rag-app
# (Use port 8001 to avoid conflict with the dev server)
```

### 9.3 Demo Flow During the Bootcamp

This is the exact sequence the instructor follows during the live session:

```
STEP 1: PULL & RUN (5 min)
    - Open Codespaces on the repo
    - Show it auto-installing dependencies
    - Run uvicorn, hit /health and /docs
    - "This is our baseline. It works. Now let's make it production-ready."

STEP 2: CI/CD DEMO (15 min)
    - Show .github/workflows/ci.yml — walk through the YAML
    - Make a small code change locally, push to main
    - Switch to GitHub → Actions tab → watch pipeline run
    - Introduce a ruff error (unused import), push → show red X
    - Fix it, push → show green check
    - "Every push is now automatically validated."

STEP 3: SECURITY DEMO (30 min)
    - Show the /query endpoint — no rate limiting
    - Run test_hammering.py → all 50 requests succeed
    - Add slowapi middleware (already in code, just uncomment/enable)
    - Restart server, re-run hammering → show 429s
    - Show /ingest without validation → upload a .txt file → crash
    - Add validation → upload .txt → clean 400 error
    - Show .env pattern, discuss prompt injection awareness

STEP 4: EVALUATION DEMO (15 min)
    - Show demo_evaluation.py — explain the 4 RAGAS metrics
    - Run with sample data → show scorecard
    - (If time) Run with --live flag against the running server
    - "Before you submit your assignment, run this on your own system."

STEP 5: DOCKER DEMO (10 min)
    - docker build → docker run → hit /health
    - Show docker-compose.yml → docker-compose up
    - "Your evaluator can now run your entire system with one command."
```

---

## 10. Ruff Configuration

Add a `ruff.toml` or section in `pyproject.toml`:

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # Line length handled separately

[tool.ruff.lint.isort]
known-first-party = ["app"]
```

---

## 11. README.md Specification

The README should be concise and professional. Sections:

1. **Title + one-line description**
2. **Quick Start** — 5 steps to get running (clone, venv, install, .env, run)
3. **API Endpoints** — table with method, path, description
4. **Architecture** — brief text + the mermaid or ASCII diagram:
   ```
   PDF → Docling Parser → [Text | Table | Image] chunks
                              ↓
                    Granite Embedding 30M
                              ↓
                        FAISS Index
                              ↓
                    Query → Retriever (top-k=4)
                              ↓
                    Granite 8B Instruct + Grounding Prompt
                              ↓
                        Answer + Sources
   ```
5. **Running Tests** — `pytest tests/ -v`
6. **Docker** — `docker build` + `docker run` commands
7. **Evaluation** — how to run RAGAS evaluation
8. **Environment Variables** — table of all env vars with defaults
9. **Tech Stack** — bullet list of key technologies

---

## 12. Critical Implementation Notes

### Things That Are Easy to Get Wrong

1. **Docling version compatibility** — Docling's API changed significantly between v1 and v2. Use `docling>=2.5.0`. The `HybridChunker` import path is `docling.chunking.HybridChunker` in v2.

2. **LangChain v0.3 imports** — Many LangChain imports moved in v0.3:
   - `from langchain_community.vectorstores import FAISS` (not `langchain.vectorstores`)
   - `from langchain_community.embeddings import HuggingFaceEmbeddings` (not `langchain.embeddings`)
   - `from langchain_community.llms import Replicate` (not `langchain.llms`)
   - Use `.invoke()` not `.run()` on chains

3. **FAISS index persistence** — `FAISS.save_local(path)` / `FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)`. The `allow_dangerous_deserialization=True` is required in recent LangChain versions.

4. **Replicate API** — The model string format for Replicate is typically `"ibm-granite/granite-3.1-8b-instruct"` but check the exact model identifier on Replicate. For the vision model, it's called via `replicate.run()` directly (not through LangChain) since LangChain's Replicate wrapper doesn't support vision inputs.

5. **FastAPI file upload** — `python-multipart` must be in requirements.txt for `UploadFile` to work. Without it, FastAPI silently fails.

6. **slowapi + FastAPI** — slowapi needs the `Request` object as the first parameter of the endpoint function. This is a common gotcha.

7. **pytest + FastAPI lifespan** — The `TestClient` context manager properly triggers the lifespan. Use `with TestClient(app) as client:` pattern.

8. **RAGAS evaluator LLM** — RAGAS 0.2.x defaults to OpenAI for its internal evaluator LLM. If participants don't have OpenAI keys, they need to configure a different evaluator. Document this.

---

## 13. What Success Looks Like

When the repo is fully built, these commands should all work:

```bash
# Start server in mock mode
MOCK_MODE=true uvicorn app.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health
# → {"status":"healthy","index_loaded":false,"document_count":0}

# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@source_documents/sample.pdf"
# → {"message":"PDF ingested successfully","filename":"sample.pdf","chunks_added":12}

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key findings?"}'
# → {"answer":"Based on the retrieved documents...","sources":[...],"query":"What are the key findings?"}

# Run tests
pytest tests/ -v
# → All 8+ tests pass

# Lint
ruff check .
# → All checks pass

# Docker
docker build -t rag-app .
docker run -p 8000:8000 -e MOCK_MODE=true rag-app
curl http://localhost:8000/health
# → {"status":"healthy",...}

# Evaluation (requires OPENAI_API_KEY or alternative evaluator config)
python evaluation/demo_evaluation.py
# → Prints RAGAS scorecard
```

---

*End of blueprint. Use this document to build the complete repo with Claude Code.*
