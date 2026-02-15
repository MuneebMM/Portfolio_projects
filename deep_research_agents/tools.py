import asyncio
import logging
import uuid
from urllib.parse import quote_plus, urlparse, parse_qs

from playwright.async_api import async_playwright
from scrapy.http import HtmlResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
import config

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=config.QDRANT_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def _ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if config.QDRANT_COLLECTION not in collections:
        qdrant_client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


async def _fetch_page(url: str) -> dict:
    """Fetch a single page using Playwright and extract content with Scrapy."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            html = await page.content()
            title = await page.title()
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            await browser.close()
            return {}
        await browser.close()

    # Use Scrapy's HtmlResponse to parse and extract text
    response = HtmlResponse(url=url, body=html, encoding="utf-8")
    # Extract main text content, skip scripts/styles
    paragraphs = response.css(
        "p::text, h1::text, h2::text, h3::text, li::text, td::text, article ::text"
    ).getall()
    content = "\n".join(line.strip() for line in paragraphs if line.strip())

    return {"content": content, "url": url, "title": title} if content else {}


async def _search_and_scrape(query: str, limit: int) -> list[dict]:
    """Search DuckDuckGo via Playwright, then scrape top result pages."""
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(search_url, timeout=15000, wait_until="domcontentloaded")
            html = await page.content()
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            await browser.close()
            return []
        await browser.close()

    # Parse search results with Scrapy
    response = HtmlResponse(url=search_url, body=html, encoding="utf-8")
    links = response.css("a.result__a::attr(href)").getall()

    # DuckDuckGo HTML wraps URLs in redirects, extract actual URLs
    result_urls = []
    for link in links:
        if "uddg=" in link:
            parsed = parse_qs(urlparse(link).query)
            if "uddg" in parsed:
                result_urls.append(parsed["uddg"][0])
        elif link.startswith("http"):
            result_urls.append(link)
        if len(result_urls) >= limit:
            break

    # Scrape each result page
    documents = []
    for url in result_urls:
        doc = await _fetch_page(url)
        if doc:
            documents.append(doc)

    return documents


def web_search(query: str) -> list[dict]:
    """Search the web and scrape results using Scrapy + Playwright."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                documents = pool.submit(
                    asyncio.run,
                    _search_and_scrape(query, config.SEARCH_RESULT_LIMIT),
                ).result()
        else:
            documents = loop.run_until_complete(
                _search_and_scrape(query, config.SEARCH_RESULT_LIMIT)
            )
    except RuntimeError:
        documents = asyncio.run(
            _search_and_scrape(query, config.SEARCH_RESULT_LIMIT)
        )
    return documents


def store_in_qdrant(documents: list[dict]) -> int:
    """Store document embeddings in Qdrant. Returns number of documents stored."""
    _ensure_collection()
    if not documents:
        return 0

    texts = [doc["content"][:4000] for doc in documents]
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "content": doc["content"][:4000],
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
            },
        )
        for doc, vector in zip(documents, vectors)
    ]

    qdrant_client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    return len(points)


def search_qdrant(query: str, limit: int = 5) -> list[dict]:
    """Retrieve relevant documents from Qdrant."""
    _ensure_collection()
    query_vector = embeddings.embed_query(query)
    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
    )
    return [
        {
            "content": point.payload.get("content", ""),
            "url": point.payload.get("url", ""),
            "title": point.payload.get("title", ""),
            "score": point.score,
        }
        for point in results.points
    ]
