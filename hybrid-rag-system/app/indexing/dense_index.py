"""Dense vector index using ChromaDB + OpenAI embeddings."""

import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import logger


class DenseIndex:
    """Manages ChromaDB vector store for dense retrieval."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Dense index initialized: collection='{settings.chroma_collection_name}'"
        )

    def add_documents(self, chunks: list[Document]) -> int:
        """Embed and store document chunks using concurrent embedding."""
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        batch_size = 500
        batch_ranges = [
            (i, min(i + batch_size, len(texts)))
            for i in range(0, len(texts), batch_size)
        ]
        total_batches = len(batch_ranges)

        # Embed all batches concurrently (I/O-bound OpenAI API calls)
        def embed_batch(args):
            idx, start, end = args
            vecs = self.embeddings.embed_documents(texts[start:end])
            return idx, start, end, vecs

        results = [None] * total_batches
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(embed_batch, (i, s, e)): i
                for i, (s, e) in enumerate(batch_ranges)
            }
            for future in as_completed(futures):
                idx, start, end, vecs = future.result()
                results[idx] = (start, end, vecs)
                logger.info(f"Embedded batch {idx + 1}/{total_batches} ({end - start} chunks)")

        # Write to ChromaDB sequentially (avoid concurrent SQLite writes)
        for start, end, vecs in results:
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=vecs,
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(f"Dense index: {len(chunks)} chunks stored")
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Query the dense index and return ranked results."""
        k = top_k or settings.dense_top_k
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # cosine similarity
                    "source": "dense",
                }
            )
        return hits

    def clear(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Dense index cleared")

    @property
    def count(self) -> int:
        return self.collection.count()
