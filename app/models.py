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
