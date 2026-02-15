"""Quick test script to run sample queries against the API."""

import httpx
import json

BASE_URL = "http://localhost:8000"

SAMPLE_QUERIES = [
    "What is retrieval augmented generation?",
    "How do transformers handle long-range dependencies?",
    "Explain the attention mechanism in neural networks",
    "What are the benefits of hybrid search?",
    "How does BM25 scoring work?",
    "What is the role of reranking in a RAG pipeline?",
    "What is ChromaDB and how does it work?",
    "What is Reciprocal Rank Fusion?",
]


def main():
    client = httpx.Client(timeout=60.0)

    # Ingest first
    print("Ingesting documents...")
    resp = client.post(f"{BASE_URL}/ingest")
    print(f"Ingest: {resp.json()}\n")

    # Run queries
    for query in SAMPLE_QUERIES:
        print(f"Q: {query}")
        resp = client.post(
            f"{BASE_URL}/query",
            json={"query": query, "top_k": 3},
        )
        data = resp.json()
        print(f"A: {data['answer'][:200]}...")
        print(f"   Sources: {[s['source'] for s in data['sources']]}")
        print(f"   Retrieved: {data['retrieval_count']} chunks\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
