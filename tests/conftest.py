"""Shared pytest fixtures."""

import os

import pytest
from fastapi.testclient import TestClient

# Force mock mode for all tests
os.environ["MOCK_MODE"] = "true"

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client with mock RAG pipeline."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_pdf_path():
    """Path to a real sample PDF from source_documents/."""
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "source_documents")
    if os.path.isdir(docs_dir):
        pdfs = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
        if pdfs:
            return os.path.join(docs_dir, pdfs[0])
    return None
