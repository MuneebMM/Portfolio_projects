import json
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import config
from tools import web_search, store_in_qdrant, search_qdrant
from database import cache_set, cache_get

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)


class ResearchState(TypedDict):
    topic: str
    research_findings: str
    analysis: str
    report: str


def searcher_node(state: ResearchState) -> dict:
    """Gathers information from the web using Firecrawl and stores in Qdrant."""
    topic = state["topic"]
    logger.info(f"Searcher: researching '{topic}'")

    # Check cache first
    cache_key = f"research:{topic}"
    cached = cache_get(cache_key)
    if cached:
        logger.info("Searcher: using cached findings")
        return {"research_findings": cached}

    # Scrape web using Scrapy + Playwright
    documents = web_search(topic)

    # Store in Qdrant for later retrieval
    if documents:
        store_in_qdrant(documents)

    # Format findings for the next agent
    findings_parts = []
    for doc in documents:
        source = f"[{doc.get('title', 'Source')}]({doc.get('url', '')})"
        findings_parts.append(f"### {source}\n{doc['content'][:3000]}")

    raw_findings = "\n\n---\n\n".join(findings_parts) if findings_parts else ""

    # Use LLM to synthesize raw findings
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are ResearchBot-X, an expert at finding and extracting high-quality, "
                    "up-to-date information from the web. Your job is to gather comprehensive, "
                    "reliable, and diverse sources on the given topic.\n"
                    "1. Extract key facts, statistics, and expert opinions.\n"
                    "2. Cover multiple perspectives and highlight any disagreements.\n"
                    "3. Organize findings in a clear, structured format.\n"
                    "4. Include References & Sources of the Content."
                )
            ),
            HumanMessage(
                content=f"Topic: {topic}\n\nRaw web findings:\n{raw_findings}\n\n"
                "Synthesize these findings into a comprehensive research summary."
            ),
        ]
    )

    findings = response.content
    cache_set(cache_key, findings)
    return {"research_findings": findings}


def analyst_node(state: ResearchState) -> dict:
    """Analyzes research findings, pulling additional context from Qdrant."""
    topic = state["topic"]
    findings = state["research_findings"]
    logger.info("Analyst: analyzing findings")

    # Pull additional relevant context from Qdrant
    qdrant_results = search_qdrant(topic, limit=3)
    extra_context = "\n".join(
        f"- {r['title']}: {r['content'][:500]}" for r in qdrant_results
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are AnalystBot-X, a critical thinker who synthesizes research findings "
                    "into actionable insights.\n"
                    "1. Identify key themes, trends, and contradictions.\n"
                    "2. Highlight the most important findings and their implications.\n"
                    "3. Suggest areas for further investigation if gaps are found.\n"
                    "4. Present analysis in a structured, easy-to-read format.\n"
                    "5. Only include reference links that were actually provided. Never hallucinate links.\n"
                    "6. If no links were provided, do not include a References section."
                )
            ),
            HumanMessage(
                content=f"Topic: {topic}\n\nResearch Findings:\n{findings}\n\n"
                f"Additional Context from Knowledge Base:\n{extra_context}\n\n"
                "Provide a thorough analysis."
            ),
        ]
    )

    analysis = response.content
    cache_set(f"analysis:{topic}", analysis)
    return {"analysis": analysis}


def writer_node(state: ResearchState) -> dict:
    """Writes the final polished report."""
    topic = state["topic"]
    analysis = state["analysis"]
    logger.info("Writer: writing report")

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are WriterBot-X, a professional technical writer. You MUST format your report using clean, well-structured Markdown.\n\n"
                    "## Formatting Rules (STRICTLY follow these):\n"
                    "- Start with a top-level `# Title` for the report\n"
                    "- Write a short **Executive Summary** paragraph right after the title\n"
                    "- Use `## Bold Section Headings` for each major section (e.g., ## Key Findings, ## Market Trends, ## Technical Analysis)\n"
                    "- Use `### Sub-headings` to break down sections further when needed\n"
                    "- Use **bold text** for key terms, metrics, and important phrases\n"
                    "- Use bullet points (`-`) for listing findings, features, or data points\n"
                    "- Use numbered lists (`1.`) for sequential steps, rankings, or prioritized items\n"
                    "- Use markdown tables (`| Column |`) when comparing data, features, or statistics\n"
                    "- Use `> blockquotes` for notable quotes or critical takeaways\n"
                    "- End with a `## Conclusion & Recommendations` section with actionable bullet points\n"
                    "- End with a `## References & Sources` section ONLY if the analyst provided actual links. Format as `- [Title](URL)`. Never hallucinate links.\n\n"
                    "## Content Rules:\n"
                    "- Be comprehensive but concise — no filler text\n"
                    "- Every section should have substantive content with specific facts, numbers, or insights\n"
                    "- Group related findings under clear thematic headings\n"
                    "- Highlight contrasting viewpoints or debates where they exist\n"
                )
            ),
            HumanMessage(
                content=f"Topic: {topic}\n\nAnalysis:\n{analysis}\n\n"
                "Write a comprehensive, well-structured research report."
            ),
        ]
    )

    return {"report": response.content}


def build_graph() -> StateGraph:
    """Build and compile the research workflow graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("searcher", searcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


research_graph = build_graph()


def run_research(topic: str) -> str:
    """Run the full research pipeline and return the final report."""
    result = research_graph.invoke(
        {"topic": topic, "research_findings": "", "analysis": "", "report": ""}
    )
    return result["report"]
