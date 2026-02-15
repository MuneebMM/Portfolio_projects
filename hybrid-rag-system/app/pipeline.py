"""RAG Pipeline: orchestrates ingestion → indexing → retrieval → reranking → generation."""

from app.ingestion import ingest_documents
from app.indexing import DenseIndex, SparseIndex
from app.retrieval import HybridRetriever
from app.reranking import CohereReranker
from app.generation import RAGGenerator
from app.core.config import settings
from app.core.logger import logger
from app.core.schemas import QueryResponse, SourceChunk, IngestResponse


class RAGPipeline:
    """End-to-end Hybrid RAG pipeline."""

    def __init__(self):
        self.dense_index = DenseIndex()
        self.sparse_index = SparseIndex()
        self.retriever = HybridRetriever(self.dense_index, self.sparse_index)
        self.reranker = CohereReranker()
        self.generator = RAGGenerator()
        self._is_indexed = self.dense_index.count > 0
        logger.info("RAG pipeline initialized")

    def ingest(self, data_dir: str | None = None) -> IngestResponse:
        """Run the full ingestion + indexing pipeline."""
        # Clear existing indices
        self.dense_index.clear()
        self.sparse_index.clear()

        # Ingest and chunk documents
        chunks = ingest_documents(data_dir)
        if not chunks:
            return IngestResponse(
                status="warning", documents_ingested=0, chunks_created=0
            )

        # Index into both stores
        dense_count = self.dense_index.add_documents(chunks)
        sparse_count = self.sparse_index.add_documents(chunks)

        self._is_indexed = True
        doc_count = len(set(c.metadata.get("source", "") for c in chunks))

        logger.info(
            f"Ingestion complete: {doc_count} docs, {dense_count} dense chunks, "
            f"{sparse_count} sparse chunks"
        )

        return IngestResponse(
            status="success",
            documents_ingested=doc_count,
            chunks_created=dense_count,
        )

    def query(self, query: str, top_k: int | None = None) -> QueryResponse:
        """Run the full RAG query pipeline."""
        if not self._is_indexed:
            return QueryResponse(
                query=query,
                answer="No documents indexed. Please run /ingest first.",
                sources=[],
                retrieval_count=0,
            )

        k = top_k or settings.rerank_top_k

        # Step 1: Hybrid retrieval
        retrieved = self.retriever.retrieve(query)

        # Step 2: Rerank
        reranked = self.reranker.rerank(query, retrieved, top_k=k)

        # Step 3: Generate
        answer = self.generator.generate(query, reranked)

        # Build response
        sources = [
            SourceChunk(
                content=r["content"][:300],
                source=r.get("metadata", {}).get("source", "unknown"),
                score=round(r.get("rerank_score", r.get("rrf_score", 0.0)), 4),
            )
            for r in reranked
        ]

        return QueryResponse(
            query=query,
            answer=answer,
            sources=sources,
            retrieval_count=len(retrieved),
        )
