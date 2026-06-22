"""
ingestion/scraper.py
Lightweight scraper using requests + BeautifulSoup.
 
Design
------
Two steps per target:
  1. Listing page  — extract article URLs using a CSS selector.
  2. Article page  — extract title + body text from the linked page.
 
Article body extraction uses a priority list of semantic containers
(article, main, [role=main]) before falling back to <p> tags.
This works for most editorial sites without needing site-specific rules.
 
Robots.txt note
---------------
Always check a site's robots.txt before scraping.  The targets in
settings.py were chosen because they permit general crawling, but you
are responsible for compliance on any site you add.
"""
 
import logging
import time
from urllib.parse import urljoin
 
import requests
from bs4 import BeautifulSoup
 
from config.settings import (
    REQUEST_DELAY_SECONDS,
    REQUEST_HEADERS,
    SCRAPE_TARGETS,
    SCRAPER_MAX_ARTICLES,
)
from storage.database import insert_article
 
logger = logging.getLogger(__name__)
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _get(url: str) -> BeautifulSoup | None:
    """GET a URL and return a BeautifulSoup object, or None on error."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None
 
 
def _extract_links(soup: BeautifulSoup, selector: str, base_url: str) -> list[str]:
    """Return absolute URLs found by CSS selector."""
    links = []
    for tag in soup.select(selector):
        href = tag.get("href", "").strip()
        if not href or href.startswith("#"):
            continue
        absolute = urljoin(base_url, href) if base_url else href
        # Skip non-HTTP links (mailto:, javascript:, etc.)
        if absolute.startswith("http"):
            links.append(absolute)
    return links
 
 
def _extract_article(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract (title, body) from an article page.
    Returns ('', '') if nothing useful is found.
    """
    # Title
    title = ""
    for selector in ("h1", "title", 'meta[property="og:title"]'):
        tag = soup.select_one(selector)
        if tag:
            title = tag.get("content", "") or tag.get_text(strip=True)
            if title:
                break
 
    # Body — look for semantic containers first
    body = ""
    for selector in ("article", "main", '[role="main"]', ".article-body", ".post-content"):
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            body = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if len(body) > 200:   # only accept if there's real content
                break
 
    # Last resort: all <p> tags on the page
    if not body:
        paragraphs = soup.find_all("p")
        body = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
 
    return title, body
 
 
# ── Per-target scraper ───────────────────────────────────────────────────────
 
def _scrape_target(target: dict) -> dict:
    """Scrape one configured target. Return {fetched, inserted}."""
    label = target["label"]
    listing_url = target["url"]
    base_url = target.get("base_url", listing_url)
    link_selector = target["link_selector"]
 
    logger.info("Scraping listing page: %s", listing_url)
    soup = _get(listing_url)
    if soup is None:
        return {"fetched": 0, "inserted": 0}
 
    links = _extract_links(soup, link_selector, base_url)
    links = list(dict.fromkeys(links))  # deduplicate while preserving order
    links = links[:SCRAPER_MAX_ARTICLES]
 
    fetched = len(links)
    inserted = 0
 
    for url in links:
        time.sleep(REQUEST_DELAY_SECONDS)
        article_soup = _get(url)
        if article_soup is None:
            continue
 
        title, body = _extract_article(article_soup)
        if not title:
            logger.debug("No title found for %s — skipping", url)
            continue
 
        if insert_article(
            url=url,
            title=title,
            body=body,
            source=label,
            topic="scraped",
        ):
            inserted += 1
            logger.debug("Inserted: %s", title[:60])
 
    logger.info("Scraper [%s] — fetched %d, inserted %d new", label, fetched, inserted)
    return {"fetched": fetched, "inserted": inserted}
 
 
# ── Public entry point ───────────────────────────────────────────────────────
 
def collect_scraped() -> dict:
    """
    Run the scraper for all configured targets.
    Returns a summary dict: {target_label: {fetched, inserted}}.
    """
    summary = {}
    for target in SCRAPE_TARGETS:
        result = _scrape_target(target)
        summary[target["label"]] = result
    return summary
 