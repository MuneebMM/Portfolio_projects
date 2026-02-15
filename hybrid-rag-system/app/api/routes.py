"""FastAPI route definitions."""

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
)
from app.core.logger import logger

router = APIRouter()

# Pipeline is injected via app state
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Ingest documents from the data directory."""
    try:
        pipeline = get_pipeline()
        result = await asyncio.to_thread(pipeline.ingest)
        return result
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Run a RAG query."""
    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
