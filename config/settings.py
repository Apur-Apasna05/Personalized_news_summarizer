"""
config/settings.py
Central configuration. All tunable values live here.
Add/remove RSS feeds and scrape targets freely.
"""
 
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# ── API keys ────────────────────────────────────────────────
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
 
# ── Storage ─────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/news.db")
 
# ── GNews settings ──────────────────────────────────────────
GNEWS_MAX_ARTICLES = int(os.getenv("GNEWS_MAX_ARTICLES", 10))
 
# GNews topics to pull (free tier supports one call per topic)
# Full list: world, nation, business, technology, entertainment,
#            sports, science, health
GNEWS_TOPICS = [
    "technology",
    "business",
    "science",
    "health",
    "sports",
]
 
# ── RSS feeds ───────────────────────────────────────────────
# Add any RSS/Atom feed URL here. Label is used for source tagging.
RSS_FEEDS = [
    # ── India News ──────────────────────────────────────────
    {"label": "The Hindu - India News", "url": "https://www.thehindu.com/news/national/feeder/default.rss"},
    {"label": "The Hindu - Telangana",  "url": "https://www.thehindu.com/news/national/telangana/feeder/default.rss"},
    {"label": "TOI - India News",      "url": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"},
    {"label": "TOI - Hyderabad",       "url": "https://timesofindia.indiatimes.com/rssfeeds/390840.cms"},
    {"label": "NDTV - India News",      "url": "https://feeds.feedburner.com/ndtvnews-india-news"},
    {"label": "Indian Express - India", "url": "https://indianexpress.com/section/india/feed/"},
    {"label": "Hindustan Times - India", "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"},
    {"label": "Economic Times - Business", "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms"},
    {"label": "News18 - India News",    "url": "https://www.news18.com/rss/india.xml"},
    
    # ── Abroad/Global News ──────────────────────────────────
    {"label": "BBC World",       "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"label": "BBC Technology",  "url": "http://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"label": "Reuters Top News","url": "https://feeds.reuters.com/reuters/topNews"},
    {"label": "NASA Breaking",   "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss"},
    {"label": "TechCrunch",      "url": "https://techcrunch.com/feed/"},
    {"label": "The Verge",       "url": "https://www.theverge.com/rss/index.xml"},
]
 
# ── Web scrape targets ───────────────────────────────────────
# Each entry needs:
#   url          - page listing article links
#   base_url     - prepended to relative hrefs
#   link_selector- CSS selector that matches <a> tags for articles
#   label        - source name stored in DB
SCRAPE_TARGETS = [
    {
        "label": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "base_url": "",           # HN links are absolute
        "link_selector": ".titleline > a",
    },
    {
        "label": "MIT Tech Review",
        "url": "https://www.technologyreview.com/",
        "base_url": "https://www.technologyreview.com",
        "link_selector": "a.teaserItem__title--3ntdm, h3 > a",
    },
]
 
# ── Scheduler ────────────────────────────────────────────────
# Fast loop  — fetch + dedup + store   (minutes)
FAST_LOOP_INTERVAL_MINUTES = 30
 
# ── Scraper behaviour ────────────────────────────────────────
SCRAPER_MAX_ARTICLES = int(os.getenv("SCRAPER_MAX_ARTICLES", 5))
 
# Polite delay between HTTP requests (seconds)
REQUEST_DELAY_SECONDS = 1.5
 
# Shared request headers (identify bot politely)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NewsBot/1.0; "
        "+https://github.com/Miss-crazy/news-summarizer)"
    )
}
 

# ── Embedding model (Phase 2 + 3) ───────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Ollama (Phase 2 + 3) ────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", 300))

# ── Processing (Phase 2) ────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", 3))
HDBSCAN_MIN_SAMPLES      = int(os.getenv("HDBSCAN_MIN_SAMPLES", 2))
PROCESSING_MIN_BATCH     = int(os.getenv("PROCESSING_MIN_BATCH", 10))

# ── ChromaDB (Phase 3) ──────────────────────────────────────
# Persistent directory where Chroma stores its data files.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
# Name of the collection inside ChromaDB that holds cluster summaries.
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "news_clusters")

# ── RAG (Phase 3) ───────────────────────────────────────────
# Number of top-k clusters retrieved per query.
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", 5))
# Minimum cosine similarity score to include a result (0–1).
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.38))

# ── Personalization (Phase 4) ────────────────────────────────
# Blend factor: 0.0 = pure semantic, 1.0 = pure preference
PERSONALIZATION_ALPHA  = float(os.getenv("PERSONALIZATION_ALPHA", 0.3))

# Weight update learning rate
WEIGHT_LEARNING_RATE   = float(os.getenv("WEIGHT_LEARNING_RATE", 0.1))

# Weight bounds (prevent runaway values)
WEIGHT_MIN             = float(os.getenv("WEIGHT_MIN", -1.0))
WEIGHT_MAX             = float(os.getenv("WEIGHT_MAX",  1.0))

# Dwell normaliser: this many seconds of reading = delta of +1.0
DWELL_NORMALISER       = float(os.getenv("DWELL_NORMALISER", 60.0))