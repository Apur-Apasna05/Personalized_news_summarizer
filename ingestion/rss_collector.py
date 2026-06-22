"""
ingestion/rss_collector.py
Parses RSS / Atom feeds with feedparser.
 
feedparser handles the messy world of feed variants so we don't have to.
For each entry we store:
  - title
  - link (as url — dedup key)
  - summary / description as body
  - feed label as source
  - published date if available
"""
 
import logging
import time
from email.utils import parsedate_to_datetime
import feedparser
 
from config.settings import REQUEST_DELAY_SECONDS, RSS_FEEDS
from storage.database import insert_article
 
logger = logging.getLogger(__name__)
 
 
def _parse_published(entry) -> str:
    """
    Try several feedparser date fields, return ISO-8601 string or ''.
    feedparser normalises most date formats but not all feeds populate them.
    """
    # feedparser provides a time_struct on published_parsed / updated_parsed
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                from datetime import datetime, timezone
                return datetime(*val[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
 
    # Fall back to raw string fields
    for field in ("published", "updated"):
        raw = getattr(entry, field, "")
        if raw:
            try:
                return parsedate_to_datetime(raw).isoformat()
            except Exception:
                return raw  # store as-is rather than lose the info
 
    return ""
 
 
def _collect_feed(feed_config: dict) -> dict:
    """Parse one RSS feed and insert new entries into the DB."""
    label = feed_config["label"]
    url = feed_config["url"]
 
    parsed = feedparser.parse(url)
 
    # feedparser sets bozo=True when the feed is malformed
    if parsed.bozo and not parsed.entries:
        logger.warning("Feed '%s' could not be parsed: %s", label, parsed.bozo_exception)
        return {"fetched": 0, "inserted": 0}
 
    fetched = len(parsed.entries)
    inserted = 0
 
    for entry in parsed.entries:
        link = getattr(entry, "link", "").strip()
        title = getattr(entry, "title", "").strip()
 
        if not link or not title:
            continue
 
        # Prefer summary > description > content (in that priority order)
        body = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )
        # content is a list of dicts in some Atom feeds
        if not body and hasattr(entry, "content"):
            body = entry.content[0].get("value", "") if entry.content else ""
 
        published_at = _parse_published(entry)
 
        if insert_article(
            url=link,
            title=title,
            body=body,
            source=label,
            topic="rss",
            published_at=published_at,
        ):
            inserted += 1
 
    logger.info("RSS [%s] — fetched %d, inserted %d new", label, fetched, inserted)
    return {"fetched": fetched, "inserted": inserted}
 
 
def collect_rss() -> dict:
    """
    Collect all configured RSS feeds.
    Returns a summary dict: {feed_label: {fetched, inserted}}.
    """
    summary = {}
    for feed in RSS_FEEDS:
        result = _collect_feed(feed)
        summary[feed["label"]] = result
        time.sleep(REQUEST_DELAY_SECONDS)
    return summary
 