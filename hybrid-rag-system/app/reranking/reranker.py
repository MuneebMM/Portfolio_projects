"""Reranking layer using Cohere Rerank API."""

import cohere

from app.core.config import settings
from app.core.logger import logger


class CohereReranker:
    """Reranks retrieval results using Cohere's rerank endpoint."""

    def __init__(self):
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = settings.cohere_rerank_model
        logger.info(f"Cohere reranker initialized: model='{self.model}'")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank retrieved chunks using Cohere API."""
        if not results:
            return []

        k = top_k or settings.rerank_top_k
        documents = [r["content"] for r in results]

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(k, len(documents)),
                return_documents=False,
            )

            reranked = []
            for item in response.results:
                result = results[item.index].copy()
                result["rerank_score"] = item.relevance_score
                reranked.append(result)

            logger.info(
                f"Reranked {len(documents)} → {len(reranked)} results"
            )
            return reranked

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}. Returning original results.")
            return results[:k]
