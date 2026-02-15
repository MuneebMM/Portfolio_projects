"""
Crawl Stripe docs and save each page as a .md file in ./data/.

Usage:
    python scripts/crawl_stripe_docs.py              # all pages
    python scripts/crawl_stripe_docs.py --max 200    # limit to 200 pages
"""

import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

SITEMAP_URL = "https://docs.stripe.com/sitemap.xml"
DATA_DIR = Path(__file__).parent.parent / "data"
DELAY = 0.3  # seconds between requests
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; hybrid-rag-crawler/1.0)"
})


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    print(f"Fetching sitemap: {sitemap_url}")
    resp = SESSION.get(sitemap_url, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    # handle sitemap index (multiple sitemaps)
    sitemap_refs = root.findall("sm:sitemap/sm:loc", ns)
    if sitemap_refs:
        urls = []
        for ref in sitemap_refs:
            urls.extend(fetch_sitemap_urls(ref.text.strip()))
        return urls
    # regular sitemap
    return [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns)]


def url_to_filename(url: str) -> str:
    """Convert a docs URL to a safe .md filename."""
    path = url.replace("https://docs.stripe.com", "").strip("/")
    if not path:
        path = "index"
    safe = re.sub(r"[^\w\-/]", "_", path).replace("/", "__")
    return f"stripe__{safe}.md"


def crawl(max_pages: int | None = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urls = fetch_sitemap_urls(SITEMAP_URL)
    print(f"Found {len(urls)} URLs in sitemap")

    if max_pages:
        urls = urls[:max_pages]
        print(f"Limiting to {max_pages} pages")

    success, skipped, failed = 0, 0, 0

    for i, url in enumerate(urls, 1):
        filename = url_to_filename(url)
        out_path = DATA_DIR / filename

        if out_path.exists():
            skipped += 1
            print(f"[{i}/{len(urls)}] SKIP (exists): {filename}")
            continue

        md_url = url.rstrip("/") + ".md"
        try:
            resp = SESSION.get(md_url, timeout=15)
            if resp.status_code == 404:
                # fall back to plain HTML if .md not available
                resp = SESSION.get(url, timeout=15)
                content = resp.text
                out_path = out_path.with_suffix(".html")
            else:
                resp.raise_for_status()
                content = resp.text

            out_path.write_text(content, encoding="utf-8")
            success += 1
            print(f"[{i}/{len(urls)}] OK  ({len(content):,} chars): {out_path.name}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(urls)}] ERR {url}: {e}")

        time.sleep(DELAY)

    print(f"\nDone. success={success}  skipped={skipped}  failed={failed}")
    print(f"Files saved to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max pages to crawl")
    args = parser.parse_args()
    crawl(max_pages=args.max)
