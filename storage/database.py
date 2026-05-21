"""
storage/database.py
SQLite layer — Phase 1 + Phase 2 schema.

Tables
------
articles
  id, url, title, body, source, topic,
  published_at, collected_at, is_processed

clusters
  id            INTEGER PK
  label         TEXT        -- human-readable topic label (from LLM)
  summary       TEXT        -- LLM-generated summary of the cluster
  article_ids   TEXT        -- JSON array of article IDs in this cluster
  embedding     BLOB        -- pickle of numpy array (summary embedding)
  created_at    TEXT
  updated_at    TEXT
  article_count INTEGER
"""

import json
import pickle
import sqlite3
import os
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


# ── Connection ────────────────────────────────────────────────────────────────

def _db_path() -> str:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        # Articles (Phase 1)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                url          TEXT    UNIQUE NOT NULL,
                title        TEXT    NOT NULL,
                body         TEXT    DEFAULT '',
                source       TEXT    DEFAULT '',
                topic        TEXT    DEFAULT '',
                published_at TEXT    DEFAULT '',
                collected_at TEXT    NOT NULL,
                is_processed INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON articles(url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_is_processed ON articles(is_processed)")

        # Clusters (Phase 2)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                label         TEXT    DEFAULT '',
                summary       TEXT    DEFAULT '',
                article_ids   TEXT    DEFAULT '[]',
                embedding     BLOB,
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL,
                article_count INTEGER DEFAULT 0
            )
        """)
    logger.info("Database initialised at %s", DB_PATH)


# ── Articles (Phase 1 — unchanged) ───────────────────────────────────────────

def insert_article(
    url: str,
    title: str,
    body: str = "",
    source: str = "",
    topic: str = "",
    published_at: str = "",
) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO articles
                (url, title, body, source, topic, published_at, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (url, title, body, source, topic, published_at, now),
        )
    return cursor.rowcount > 0


def fetch_unprocessed(limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM articles
            WHERE is_processed = 0
            ORDER BY collected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_processed(article_ids: list[int]) -> None:
    if not article_ids:
        return
    placeholders = ",".join("?" * len(article_ids))
    with get_connection() as conn:
        conn.execute(
            f"UPDATE articles SET is_processed = 1 WHERE id IN ({placeholders})",
            article_ids,
        )


def article_count() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE is_processed = 0"
        ).fetchone()[0]
    return {"total": total, "unprocessed": unprocessed}


# ── Clusters (Phase 2) ────────────────────────────────────────────────────────

def upsert_cluster(
    label: str,
    summary: str,
    article_ids: list[int],
    embedding=None,
) -> int:
    """Insert a new cluster. Returns the new cluster id."""
    now = datetime.now(timezone.utc).isoformat()
    embedding_blob = pickle.dumps(embedding) if embedding is not None else None

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO clusters
                (label, summary, article_ids, embedding,
                 created_at, updated_at, article_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                label,
                summary,
                json.dumps(article_ids),
                embedding_blob,
                now,
                now,
                len(article_ids),
            ),
        )
        return cursor.lastrowid


def fetch_all_clusters() -> list[dict]:
    """Return all clusters without embeddings (for display / RAG)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, label, summary, article_ids,
                   article_count, created_at, updated_at
            FROM clusters
            ORDER BY created_at DESC
            """
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["article_ids"] = json.loads(d["article_ids"])
        result.append(d)
    return result


def fetch_cluster_with_embedding(cluster_id: int) -> dict | None:
    """Return one cluster including its embedding numpy array."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["article_ids"] = json.loads(d["article_ids"])
    if d["embedding"]:
        d["embedding"] = pickle.loads(d["embedding"])
    return d


def fetch_all_cluster_embeddings() -> list[dict]:
    """Return id + label + summary + embedding for all clusters (RAG)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, label, summary, embedding FROM clusters"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d["embedding"]:
            d["embedding"] = pickle.loads(d["embedding"])
        result.append(d)
    return result


def cluster_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]


def clear_clusters() -> None:
    """Wipe clusters table — called before a full re-cluster run."""
    with get_connection() as conn:
        conn.execute("DELETE FROM clusters")
    logger.info("Clusters table cleared for re-clustering.")