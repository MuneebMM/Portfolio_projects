"""Sparse retrieval index using BM25."""

import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import logger


class SparseIndex:
    """BM25-based sparse retrieval index."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def add_documents(self, chunks: list[Document]) -> int:
        """Build BM25 index from document chunks."""
        if not chunks:
            return 0

        self.documents = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.page_content) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Sparse index: {len(chunks)} chunks indexed")
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Query the BM25 index and return ranked results."""
        if self.bm25 is None:
            logger.warning("Sparse index is empty")
            return []

        k = top_k or settings.sparse_top_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        hits = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                hits.append(
                    {
                        "content": self.documents[idx].page_content,
                        "metadata": self.documents[idx].metadata,
                        "score": float(scores[idx]),
                        "source": "sparse",
                    }
                )
        return hits

    def clear(self):
        """Reset the sparse index."""
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        logger.info("Sparse index cleared")

    @property
    def count(self) -> int:
        return len(self.documents)
