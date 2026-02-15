"""Basic evaluation script for the Hybrid RAG system."""

import json
import time
import httpx

BASE_URL = "http://localhost:8000"

# Test queries with expected keywords in the answer
EVAL_CASES = [
    {
        "query": "What is retrieval augmented generation?",
        "expected_keywords": ["retriever", "generator", "knowledge", "documents"],
        "category": "definition",
    },
    {
        "query": "How does the self-attention mechanism work in transformers?",
        "expected_keywords": ["attention", "position", "sequence", "heads"],
        "category": "mechanism",
    },
    {
        "query": "What are the benefits of hybrid search over pure dense retrieval?",
        "expected_keywords": ["sparse", "dense", "keyword", "semantic"],
        "category": "comparison",
    },
    {
        "query": "How does BM25 scoring work?",
        "expected_keywords": ["term frequency", "document", "idf", "scoring"],
        "category": "algorithm",
    },
    {
        "query": "Why is reranking important in RAG systems?",
        "expected_keywords": ["precision", "relevance", "quality", "candidates"],
        "category": "importance",
    },
    {
        "query": "What is ChromaDB used for?",
        "expected_keywords": ["vector", "embedding", "search", "store"],
        "category": "tool",
    },
    {
        "query": "Explain the difference between dense and sparse retrieval",
        "expected_keywords": ["vector", "keyword", "semantic", "bm25"],
        "category": "comparison",
    },
    {
        "query": "What is Reciprocal Rank Fusion?",
        "expected_keywords": ["rank", "merge", "results", "retrieval"],
        "category": "algorithm",
    },
]


def run_evaluation():
    """Run eval suite against the running RAG API."""
    client = httpx.Client(timeout=60.0)

    # Step 1: Ingest
    print("=" * 60)
    print("HYBRID RAG EVALUATION")
    print("=" * 60)
    print("\n[1/3] Ingesting documents...")

    try:
        resp = client.post(f"{BASE_URL}/ingest")
        resp.raise_for_status()
        ingest_result = resp.json()
        print(f"  ✓ Ingested {ingest_result['documents_ingested']} docs, "
              f"{ingest_result['chunks_created']} chunks\n")
    except Exception as e:
        print(f"  ✗ Ingestion failed: {e}")
        print("  Make sure the server is running: python -m app.main")
        return

    # Step 2: Run queries
    print("[2/3] Running evaluation queries...\n")
    results = []
    total_latency = 0

    for i, case in enumerate(EVAL_CASES, 1):
        start = time.time()
        try:
            resp = client.post(
                f"{BASE_URL}/query",
                json={"query": case["query"], "top_k": 5},
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - start
            total_latency += latency

            # Check keyword coverage
            answer_lower = data["answer"].lower()
            found = [kw for kw in case["expected_keywords"] if kw.lower() in answer_lower]
            coverage = len(found) / len(case["expected_keywords"])
            has_sources = len(data["sources"]) > 0

            result = {
                "query": case["query"],
                "category": case["category"],
                "keyword_coverage": coverage,
                "keywords_found": found,
                "keywords_missing": [k for k in case["expected_keywords"] if k.lower() not in answer_lower],
                "has_sources": has_sources,
                "source_count": len(data["sources"]),
                "retrieval_count": data["retrieval_count"],
                "latency_s": round(latency, 2),
                "answer_length": len(data["answer"]),
                "pass": coverage >= 0.5 and has_sources,
            }
            results.append(result)

            status = "✓ PASS" if result["pass"] else "✗ FAIL"
            print(f"  [{i}/{len(EVAL_CASES)}] {status} | coverage={coverage:.0%} | "
                  f"latency={latency:.2f}s | sources={len(data['sources'])}")
            print(f"       Q: {case['query'][:60]}")
            if result["keywords_missing"]:
                print(f"       Missing: {result['keywords_missing']}")
            print()

        except Exception as e:
            results.append({
                "query": case["query"],
                "category": case["category"],
                "error": str(e),
                "pass": False,
            })
            print(f"  [{i}/{len(EVAL_CASES)}] ✗ ERROR: {e}\n")

    # Step 3: Summary
    print("[3/3] Evaluation Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    avg_coverage = sum(r.get("keyword_coverage", 0) for r in results) / total
    avg_latency = total_latency / total if total > 0 else 0

    print(f"  Tests Passed:      {passed}/{total} ({passed/total:.0%})")
    print(f"  Avg Keyword Cover: {avg_coverage:.0%}")
    print(f"  Avg Latency:       {avg_latency:.2f}s")
    print(f"  Total Time:        {total_latency:.2f}s")
    print("=" * 60)

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    run_evaluation()
