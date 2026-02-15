# Hybrid RAG System

> A production-grade **Retrieval-Augmented Generation** pipeline that combines dense vector search, sparse BM25 keyword search, and Cohere neural reranking to deliver highly accurate, context-grounded answers from your own documents.

---

## What Is This?

Standard RAG systems rely on a single retrieval method — usually a vector similarity search. This works well for semantic queries but fails on keyword-specific or lexical queries. The **Hybrid RAG System** solves this by running **two complementary retrieval strategies in parallel**, fusing their results using **Reciprocal Rank Fusion (RRF)**, and then applying a **Cohere neural reranker** to select only the most relevant chunks before generation.

The result is a system that consistently outperforms single-retrieval RAG in answer relevance, factual accuracy, and coverage — proven to work on real documentation sets of 3000+ chunks.

---

## Architecture

```
Documents (PDF, DOCX, HTML, JSON, TXT, MD)
        │
        ▼
  Document Loader  ──►  Text Chunker (512 tokens, 50 overlap)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  Dense Index                           Sparse Index
  (ChromaDB +                           (BM25Okapi —
  OpenAI text-embedding-3-small)        rank-bm25)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
            Reciprocal Rank Fusion
            (merges & deduplicates)
                       │
                       ▼
            Cohere Reranker
            (rerank-english-v3.0)
            17–20 candidates → top 5
                       │
                       ▼
            OpenAI GPT (ChatOpenAI)
            context-grounded answer generation
                       │
                       ▼
               Final Answer + Sources
```

---

## Key Features

### Multi-Format Document Support
Ingest documents in any of the following formats without any preprocessing:

| Format | Extension | Parser |
|--------|-----------|--------|
| Plain text | `.txt` | Native |
| Markdown | `.md` | Native |
| PDF | `.pdf` | pypdf |
| Word Document | `.docx`, `.doc` | python-docx |
| HTML | `.html`, `.htm` | BeautifulSoup + lxml |
| JSON | `.json` | stdlib json |

### Hybrid Retrieval
- **Dense retrieval** — OpenAI `text-embedding-3-small` embeddings stored in ChromaDB, captures semantic similarity
- **Sparse retrieval** — BM25 keyword matching via `rank-bm25`, captures exact term matches
- **RRF fusion** — merges and deduplicates both result sets without requiring score normalization

### Neural Reranking
- Cohere `rerank-english-v3.0` re-scores 17–20 fused candidates and selects the top 5
- Proven to improve answer quality by surfacing contextually relevant chunks that rank lower in raw retrieval
- Every query response includes Cohere relevance scores (0–1) alongside source citations

### Optimized Ingestion Pipeline
- Embeddings generated in **concurrent batches** (5 parallel OpenAI API calls) via `ThreadPoolExecutor`
- Batch size of 500 chunks reduces API round-trips from 33 → 7
- Ingestion runs via `asyncio.to_thread` — the server stays fully responsive during indexing
- Result: **~8× faster ingestion** (67s → ~8s for 3282 chunks)

### Production-Ready API
- FastAPI backend with Pydantic v2 request/response validation
- `/health`, `/ingest`, `/query` endpoints
- Full OpenAPI / Swagger docs auto-generated at `/docs`
- Non-blocking async architecture

### Streamlit Chat UI
- Real-time chat interface with full conversation history
- Expandable sources panel per answer showing document name, Cohere relevance score, and content preview
- One-click document ingestion with live status feedback
- Server health indicator in sidebar

---

## Use Cases

### 1. Internal Knowledge Base Q&A
Drop your company's policy documents, wikis, SOPs, or handbooks into the `data/` folder. Ask natural language questions and get precise, source-cited answers — no more manually searching through PDFs.

### 2. API / Technical Documentation Assistant
Ingest API docs, SDK references, or changelogs. Developers can ask *"How do I authenticate with the API?"* or *"What are the rate limits for the embeddings endpoint?"* and get direct answers with source citations.

### 3. Legal & Compliance Document Search
Upload contracts, compliance guidelines, or regulatory documents. The hybrid retrieval ensures both semantic queries (*"what are my obligations under this clause"*) and keyword queries (*"section 4.2 liability"*) both return accurate results.

### 4. Research Paper Summarization & QA
Ingest academic papers or technical reports. Ask cross-document questions and get synthesized answers grounded in the actual text.

### 5. Customer Support Knowledge Base
Build an internal assistant that answers support queries from your product documentation, reducing ticket volume and onboarding time for new support agents.

### 6. Personal AI Assistant over Your Own Files
Point the system at any collection of notes, ebooks, or documents. Query your own knowledge base conversationally through the chat UI.

---

## Why Hybrid RAG Outperforms Standard RAG

| Scenario | Dense Only | Sparse Only | Hybrid + Rerank |
|---|---|---|---|
| Semantic / conceptual query | Good | Misses | Best |
| Keyword / code / exact query | Misses | Good | Best |
| Ambiguous query | Partial | Partial | Best |
| Multi-document answer | Partial | Misses | Best |

The reranker acts as a **second-pass quality filter** — it scores candidates by true semantic relevance to the query, not just vector proximity. This catches relevant chunks that ranked low in initial retrieval and removes high-ranking but irrelevant ones.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT (via LangChain) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (persistent) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Reranking | Cohere `rerank-english-v3.0` |
| Document Parsing | pypdf, python-docx, BeautifulSoup, lxml |
| UI | Streamlit |
| Configuration | Pydantic Settings v2 + `.env` |
| Runtime | Python 3.13 |

---

## Project Structure

```
hybrid-rag-system/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints (/health, /ingest, /query)
│   ├── core/
│   │   ├── config.py              # Pydantic settings loaded from .env
│   │   ├── logger.py              # Structured logging (UTF-8 safe on Windows)
│   │   └── schemas.py             # Request/response Pydantic models
│   ├── generation/
│   │   └── generator.py           # LangChain + OpenAI answer generation
│   ├── indexing/
│   │   ├── dense_index.py         # ChromaDB + concurrent embedding batches
│   │   └── sparse_index.py        # BM25 index
│   ├── ingestion/
│   │   └── ingest.py              # Multi-format document loader + text chunker
│   ├── retrieval/
│   │   └── hybrid_retriever.py    # RRF fusion of dense + sparse results
│   ├── reranking/
│   │   └── reranker.py            # Cohere rerank API integration
│   ├── pipeline.py                # Orchestrates the full pipeline
│   └── main.py                    # FastAPI app entry point
├── data/                          # Place your documents here
├── scripts/
│   ├── crawl_stripe_docs.py       # Web crawler for documentation sites
│   ├── test_queries.py
│   └── evaluate.py
├── ui.py                          # Streamlit chat interface
├── requirements.txt
├── pyproject.toml
├── .env                           # API keys and configuration
└── docker-compose.yml
```

---

## Setup & Installation

### Prerequisites
- Python 3.13+
- OpenAI API key — [platform.openai.com](https://platform.openai.com)
- Cohere API key — [cohere.com](https://cohere.com)

### 1. Clone the repository

```bash
git clone https://github.com/MuneebMM/hybrid-rag-system.git
cd hybrid-rag-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (Git Bash)
source .venv/Scripts/activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `uv` (significantly faster):

```bash
uv pip install -r requirements.txt
```

### 4. Configure environment variables

Fill in your API keys in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

COHERE_API_KEY=...
COHERE_RERANK_MODEL=rerank-english-v3.0

CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=hybrid_rag

DENSE_TOP_K=10
SPARSE_TOP_K=10
RERANK_TOP_K=5

CHUNK_SIZE=512
CHUNK_OVERLAP=50

DATA_DIR=./data
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 5. Add your documents

Place any supported files into the `data/` directory:

```
data/
├── documentation.pdf
├── report.docx
├── knowledge-base.html
├── config.json
└── notes.md
```

### 6. Start the API server

```bash
python -m app.main
```

API available at `http://localhost:8000` — Swagger UI at `http://localhost:8000/docs`

### 7. Launch the Streamlit UI

Open a second terminal:

```bash
streamlit run ui.py
```

Chat UI opens automatically at `http://localhost:8501`

### 8. Ingest and query

1. Click **Ingest Documents** in the Streamlit sidebar
2. Wait for confirmation (shows document count and chunk count)
3. Type your question in the chat input
4. View the answer and expand **Sources** to see which chunks were used

---

## API Reference

### `GET /health`
```json
{ "status": "healthy", "version": "1.0.0" }
```

### `POST /ingest`
Triggers document ingestion from the configured `DATA_DIR`. Non-blocking — server remains responsive.

```json
{
  "status": "success",
  "documents_ingested": 1,
  "chunks_created": 3282
}
```

### `POST /query`

**Request:**
```json
{
  "query": "How does function calling work?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "How does function calling work?",
  "answer": "Function calling allows you to describe functions to the model...",
  "sources": [
    {
      "content": "You can describe functions and have the model intelligently...",
      "source": "openai-api-docs-complete.html",
      "score": 0.9821
    }
  ],
  "retrieval_count": 17
}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `COHERE_API_KEY` | required | Cohere API key |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Reranking model |
| `DENSE_TOP_K` | `10` | Candidates retrieved from dense index |
| `SPARSE_TOP_K` | `10` | Candidates retrieved from sparse index |
| `RERANK_TOP_K` | `5` | Final results after reranking |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `DATA_DIR` | `./data` | Document input directory |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistent storage path |

---

## Performance

| Metric | Result |
|---|---|
| Ingestion speed | 3282 chunks in ~8 seconds |
| Embedding concurrency | 5 parallel API calls |
| Query latency | 3–6 seconds end-to-end |
| Retrieval candidates | 17–20 fused results per query |
| Final context chunks | Top 5 after Cohere reranking |
| Supported file formats | 8 (txt, md, pdf, docx, doc, html, htm, json) |

---

## Author

**MuneebMM**

Built as a portfolio project demonstrating production-grade AI system design — combining vector databases, hybrid search strategies, neural reranking, and LLM-powered generation into a cohesive, deployable pipeline.

---

*If you found this useful, feel free to star the repository.*
