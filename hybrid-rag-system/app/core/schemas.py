"""Pydantic schemas for request/response models."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query string")
    top_k: int = Field(5, description="Number of final results after reranking")


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]
    retrieval_count: int


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
