"""
ingestion/gnews_collector.py
Fetches articles from the GNews API for each configured topic.

Free tier limits
----------------
  - 10 articles per request
  - 100 requests / day
  - Only title + description in response (no full body)

We store description as `body` for now; phase 2 summarization
will work on what we have. If you upgrade to a paid key, set
GNEWS_EXPAND=true in .env and the collector will fetch full content.
"""

import logging
import time

import requests

from config.settings import (
    GNEWS_API_KEY,
    GNEWS_MAX_ARTICLES,
    GNEWS_TOPICS,
    REQUEST_DELAY_SECONDS,
)
from storage.database import insert_article

logger = logging.getLogger(__name__)

GNEWS_ENDPOINT = "https://gnews.io/api/v4/top-headlines"


def _fetch_topic(topic: str) -> list[dict]:
    """Call GNews for one topic; return list of raw article dicts."""
    if not GNEWS_API_KEY or GNEWS_API_KEY == "your_gnews_api_key_here":
        logger.warning("GNEWS_API_KEY not set — skipping GNews collector.")
        return []

    params = {
        "token": GNEWS_API_KEY,
        "topic": topic,
        "lang": "en",
        "max": GNEWS_MAX_ARTICLES,
    }
    try:
        resp = requests.get(GNEWS_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except requests.RequestException as exc:
        logger.error("GNews request failed for topic '%s': %s", topic, exc)
        return []


def collect_gnews() -> dict:
    """
    Iterate over all configured topics, fetch articles, persist new ones.
    Returns a summary dict: {topic: {fetched, inserted}}.
    """
    summary = {}
    for topic in GNEWS_TOPICS:
        articles = _fetch_topic(topic)
        inserted = 0
        for art in articles:
            url = art.get("url", "").strip()
            title = art.get("title", "").strip()
            if not url or not title:
                continue

            body = art.get("description") or art.get("content") or ""
            source = art.get("source", {}).get("name", "GNews")
            published_at = art.get("publishedAt", "")

            if insert_article(
                url=url,
                title=title,
                body=body,
                source=source,
                topic=topic,
                published_at=published_at,
            ):
                inserted += 1

        summary[topic] = {"fetched": len(articles), "inserted": inserted}
        logger.info(
            "GNews [%s] — fetched %d, inserted %d new",
            topic, len(articles), inserted,
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    return summary