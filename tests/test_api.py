"""Tests for FastAPI endpoints (all run in mock mode)."""

import io


def test_health_endpoint(client):
    """GET /health returns 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    """GET /health response has required fields."""
    data = client.get("/health").json()
    assert "status" in data
    assert "index_loaded" in data
    assert "document_count" in data
    assert data["status"] == "healthy"


def test_query_no_index(client):
    """POST /query before ingestion returns 400."""
    response = client.post("/query", json={"question": "What is this about?"})
    assert response.status_code == 400
    assert "No documents" in response.json()["detail"]


def test_query_empty_string(client):
    """POST /query with empty question returns 422."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_too_long(client):
    """POST /query with >1000 char question returns 422."""
    long_question = "x" * 1001
    response = client.post("/query", json={"question": long_question})
    assert response.status_code == 422


def test_query_missing_field(client):
    """POST /query with no question field returns 422."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_ingest_non_pdf(client):
    """POST /ingest with a .txt file returns 400."""
    fake_file = io.BytesIO(b"hello world")
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_valid_pdf(client, sample_pdf_path):
    """POST /ingest with a real PDF returns 200 with chunk count."""
    if sample_pdf_path is None:
        # No sample PDFs available — create a minimal one
        pdf_bytes = b"%PDF-1.4 minimal"
        response = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    else:
        with open(sample_pdf_path, "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": (sample_pdf_path.split("/")[-1], f, "application/pdf")},
            )
    assert response.status_code == 200
    data = response.json()
    assert "chunks_added" in data
    assert data["chunks_added"] > 0


def test_query_after_ingest(client):
    """POST /query after ingestion returns a valid response."""
    # First ingest a mock PDF
    pdf_bytes = b"%PDF-1.4 minimal"
    client.post(
        "/ingest",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    # Now query
    response = client.post("/query", json={"question": "What are the key findings?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "query" in data
    assert data["query"] == "What are the key findings?"


def test_docs_endpoint(client):
    """GET /docs returns 200 (Swagger UI)."""
    response = client.get("/docs")
    assert response.status_code == 200
