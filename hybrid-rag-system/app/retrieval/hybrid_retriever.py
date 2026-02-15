"""Hybrid retriever: merges dense and sparse results with RRF."""

from app.indexing.dense_index import DenseIndex
from app.indexing.sparse_index import SparseIndex
from app.core.logger import logger


class HybridRetriever:
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion."""

    def __init__(self, dense_index: DenseIndex, sparse_index: SparseIndex):
        self.dense_index = dense_index
        self.sparse_index = sparse_index

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: list[list[dict]], k: int = 60
    ) -> list[dict]:
        """Merge multiple ranked lists using RRF.

        RRF score = sum(1 / (k + rank)) across all lists.
        """
        scores: dict[str, float] = {}
        content_map: dict[str, dict] = {}

        for results in result_lists:
            for rank, hit in enumerate(results):
                key = hit["content"][:200]  # Use content prefix as key
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                if key not in content_map:
                    content_map[key] = hit

        # Sort by fused score
        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused_results = []
        for key in ranked_keys:
            result = content_map[key].copy()
            result["rrf_score"] = scores[key]
            fused_results.append(result)

        return fused_results

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval: dense + sparse → RRF fusion."""
        logger.info(f"Hybrid retrieval for: '{query[:80]}...'")

        dense_results = self.dense_index.search(query, top_k=dense_top_k)
        logger.info(f"Dense results: {len(dense_results)}")

        sparse_results = self.sparse_index.search(query, top_k=sparse_top_k)
        logger.info(f"Sparse results: {len(sparse_results)}")

        fused = self._reciprocal_rank_fusion([dense_results, sparse_results])
        logger.info(f"Fused results: {len(fused)}")

        return fused
