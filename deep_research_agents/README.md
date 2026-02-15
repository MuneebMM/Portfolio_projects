# Deep Researcher Agent

![Demo](./assets/Demo.png)

A multi-stage AI-powered research workflow agent that automates comprehensive web research, analysis, and report generation using LangGraph, GPT-5-mini, Scrapy + Playwright, Qdrant, PostgreSQL, and Redis.

## Features

- **LangGraph Pipeline**: Searcher → Analyst → Writer workflow built as a compiled state graph
- **Web Scraping**: Scrapy + Playwright for JavaScript-rendered page scraping via DuckDuckGo search
- **Vector Storage**: Qdrant for storing and retrieving research document embeddings
- **AI-Powered Analysis**: GPT-5-mini for intelligent synthesis and report writing
- **Report Persistence**: PostgreSQL for saving completed research reports
- **Caching**: Redis for caching intermediate results (findings, analysis)
- **FastAPI Backend**: REST API with streaming research endpoint
- **Streamlit Web UI**: Interactive frontend with research history
- **LangSmith Observability**: Full tracing of the LangGraph pipeline
- **Docker Compose**: One-command deployment of all 5 services

## How It Works

![Workflow](./assets/GIF.gif)

1. **Searcher Node**: Searches the web via DuckDuckGo, scrapes result pages using Playwright, parses content with Scrapy, stores embeddings in Qdrant, and synthesizes findings with GPT-5-mini.
2. **Analyst Node**: Pulls additional context from Qdrant, analyzes research findings, identifies key themes, trends, and contradictions.
3. **Writer Node**: Produces a well-structured Markdown report with bold section headings, tables, bullet points, and actionable recommendations.

> **Architecture:**
>
> ```
> User (Streamlit) → FastAPI API → LangGraph Pipeline
>                                     ├─ Searcher (GPT-5-mini + Scrapy/Playwright → Qdrant)
>                                     ├─ Analyst (GPT-5-mini + Qdrant retrieval)
>                                     └─ Writer (GPT-5-mini → Structured Report)
>                                          ↓
>                                     PostgreSQL (save report) + Redis (cache)
>                                          ↓
>                                     LangSmith (observability)
> ```

## Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) and Docker Compose
- [uv](https://github.com/astral-sh/uv) for dependency management (optional, for local dev)
- API keys for [OpenAI](https://platform.openai.com/) and optionally [LangSmith](https://smith.langchain.com/)

## Installation

### Option 1: Docker Compose (Recommended)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MuneebMM/Deep_Research_Agents.git
   cd Deep_Research_Agents
   ```

2. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start all services:**

   ```bash
   docker-compose up
   ```

   This starts 5 services: FastAPI (port 8000), Streamlit (port 8501), Qdrant (port 6333), PostgreSQL (port 5432), and Redis (port 6379).

### Option 2: Local Development

1. **Clone and navigate:**

   ```bash
   git clone https://github.com/MuneebMM/Deep_Research_Agents.git
   cd Deep_Research_Agents
   ```

2. **Create virtual environment and install dependencies:**

   ```bash
   uv venv .venv
   uv pip install -r requirements.txt
   ```

3. **Install Playwright browsers:**

   ```bash
   playwright install chromium
   ```

4. **Start infrastructure services:**

   ```bash
   docker-compose up qdrant postgres redis -d
   ```

5. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. **Run the API and frontend:**

   ```bash
   # Terminal 1 - FastAPI backend
   python -m uvicorn api:app --host 0.0.0.0 --port 8000

   # Terminal 2 - Streamlit frontend
   streamlit run app.py
   ```

## Environment Setup

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
SEARCH_RESULT_LIMIT=5
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://user:pass@127.0.0.1:5432/deep_researcher
REDIS_URL=redis://localhost:6379
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=deep-researcher-agent
```

## Usage

### Web Interface

Open [http://localhost:8501](http://localhost:8501) in your browser. Enter a research topic and the agent will stream the pipeline progress (Searching → Analyzing → Writing) before displaying the final report.

### API

```bash
# Start a research task (streaming response)
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI trends 2026"}'

# List saved reports
curl http://localhost:8000/reports

# Get a specific report
curl http://localhost:8000/reports/1
```

## Project Structure

```
deep_researcher_agent/
├── api.py              # FastAPI backend with REST endpoints
├── app.py              # Streamlit web interface
├── graph.py            # LangGraph workflow (Searcher → Analyst → Writer)
├── tools.py            # Scrapy + Playwright scraping & Qdrant vector tools
├── database.py         # PostgreSQL models & Redis cache helpers
├── config.py           # Environment configuration & LangSmith setup
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
├── Dockerfile          # Container image for API/frontend
├── docker-compose.yml  # Multi-service orchestration
├── .env                # Environment variables (not committed)
├── assets/             # Static assets (images)
└── README.md           # This file
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **AI Framework** | LangChain + LangGraph |
| **LLM** | GPT-5-mini (OpenAI) |
| **Web Scraping** | Scrapy + Playwright |
| **Vector Database** | Qdrant |
| **Relational Database** | PostgreSQL |
| **Cache** | Redis |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit |
| **Observability** | LangSmith |
| **Containerization** | Docker Compose |

## Acknowledgments

- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Scrapy](https://scrapy.org/) & [Playwright](https://playwright.dev/) for web scraping
- [Qdrant](https://qdrant.tech/) for vector search
- [OpenAI](https://openai.com/) for GPT-5-mini
- [Streamlit](https://streamlit.io/) for the web interface
- [LangSmith](https://smith.langchain.com/) for observability

## Author

Developed by [MuneebMM](https://github.com/MuneebMM)
