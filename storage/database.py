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

import urllib.parse
from config.settings import DB_PATH, DATABASE_URL

logger = logging.getLogger(__name__)

IS_POSTGRES = bool(DATABASE_URL)


# ── Connection ────────────────────────────────────────────────────────────────

def _db_path() -> str:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return DB_PATH


def parse_db_url(url_str: str):
    """Robust parser for database URLs that handles special characters in password."""
    # Strip prefix
    if url_str.startswith("postgresql://"):
        s = url_str[len("postgresql://"):]
    elif url_str.startswith("postgres://"):
        s = url_str[len("postgres://"):]
    else:
        s = url_str
        
    # Split credentials and host at the last '@'
    if "@" in s:
        creds, host_part = s.rsplit("@", 1)
    else:
        creds = ""
        host_part = s
        
    # Parse credentials
    username = ""
    password = ""
    if creds:
        if ":" in creds:
            username, password = creds.split(":", 1)
            username = urllib.parse.unquote(username)
            password = urllib.parse.unquote(password)
        else:
            username = urllib.parse.unquote(creds)
            
    # Parse host, port, database
    if "/" in host_part:
        netloc, db_part = host_part.split("/", 1)
    else:
        netloc = host_part
        db_part = ""
        
    if "?" in db_part:
        database = db_part.split("?", 1)[0]
    else:
        database = db_part
        
    if ":" in netloc:
        hostname, port_str = netloc.split(":", 1)
        port = int(port_str)
    else:
        hostname = netloc
        port = 5432
        
    return username, password, hostname, port, database


@contextmanager
def get_connection():
    if IS_POSTGRES:
        import pg8000
        import ssl
        
        username, password, hostname, port, database = parse_db_url(DATABASE_URL)
        
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        
        conn = pg8000.dbapi.connect(
            user=username,
            password=password,
            host=hostname,
            port=port,
            database=database,
            ssl_context=ssl_ctx
        )
    else:
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


def db_execute(cursor, sql: str, params=()):
    if IS_POSTGRES:
        sql = sql.replace("?", "%s")
    cursor.execute(sql, params)
    return cursor


def fetch_rows(cursor) -> list[dict]:
    if not cursor.description:
        return []
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_row(cursor) -> dict | None:
    row = cursor.fetchone()
    if not row:
        return None
    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist — articles, clusters, user profiles."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if IS_POSTGRES:
            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS articles (
                    id           SERIAL PRIMARY KEY,
                    url          VARCHAR(1024) UNIQUE NOT NULL,
                    title        TEXT    NOT NULL,
                    body         TEXT    DEFAULT '',
                    source       TEXT    DEFAULT '',
                    topic        TEXT    DEFAULT '',
                    published_at TEXT    DEFAULT '',
                    collected_at TEXT    NOT NULL,
                    is_processed INTEGER DEFAULT 0
                )
            """)
            db_execute(cursor, "CREATE INDEX IF NOT EXISTS idx_articles_is_processed ON articles(is_processed)")

            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS clusters (
                    id            SERIAL PRIMARY KEY,
                    label         TEXT    DEFAULT '',
                    summary       TEXT    DEFAULT '',
                    article_ids   TEXT    DEFAULT '[]',
                    embedding     BYTEA,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    article_count INTEGER DEFAULT 0
                )
            """)
        else:
            db_execute(cursor, """
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
            db_execute(cursor, "CREATE INDEX IF NOT EXISTS idx_url ON articles(url)")
            db_execute(cursor, "CREATE INDEX IF NOT EXISTS idx_is_processed ON articles(is_processed)")

            db_execute(cursor, """
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
    logger.info("Database initialised at %s (Postgres: %s)", DB_PATH, IS_POSTGRES)
    from storage.user_profiles import init_user_tables
    init_user_tables()


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
        cursor = conn.cursor()
        if IS_POSTGRES:
            db_execute(cursor,
                """
                INSERT INTO articles
                    (url, title, body, source, topic, published_at, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url) DO NOTHING
                """,
                (url, title, body, source, topic, published_at, now),
            )
        else:
            db_execute(cursor,
                """
                INSERT OR IGNORE INTO articles
                    (url, title, body, source, topic, published_at, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, title, body, source, topic, published_at, now),
            )
        rowcount = cursor.rowcount
    return rowcount > 0


def fetch_unprocessed(limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor,
            """
            SELECT * FROM articles
            WHERE is_processed = 0
            ORDER BY collected_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = fetch_rows(cursor)
    return rows


def mark_processed(article_ids: list[int]) -> None:
    if not article_ids:
        return
    placeholders = ",".join("?" * len(article_ids))
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor,
            f"UPDATE articles SET is_processed = 1 WHERE id IN ({placeholders})",
            article_ids,
        )


def article_count() -> dict:
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]
        db_execute(cursor, "SELECT COUNT(*) FROM articles WHERE is_processed = 0")
        unprocessed = cursor.fetchone()[0]
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
        cursor = conn.cursor()
        if IS_POSTGRES:
            db_execute(cursor,
                """
                INSERT INTO clusters
                    (label, summary, article_ids, embedding,
                     created_at, updated_at, article_count)
                VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id
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
            return cursor.fetchone()[0]
        else:
            db_execute(cursor,
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
        cursor = conn.cursor()
        db_execute(cursor,
            """
            SELECT id, label, summary, article_ids,
                   article_count, created_at, updated_at
            FROM clusters
            ORDER BY created_at DESC
            """
        )
        rows = fetch_rows(cursor)
    result = []
    for d in rows:
        if isinstance(d["article_ids"], str):
            d["article_ids"] = json.loads(d["article_ids"])
        result.append(d)
    return result


def fetch_cluster_with_embedding(cluster_id: int) -> dict | None:
    """Return one cluster including its embedding numpy array."""
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "SELECT * FROM clusters WHERE id = ?", (cluster_id,))
        d = fetch_row(cursor)
    if not d:
        return None
    if isinstance(d["article_ids"], str):
        d["article_ids"] = json.loads(d["article_ids"])
    if d["embedding"]:
        emb_data = d["embedding"]
        if isinstance(emb_data, memoryview):
            emb_data = emb_data.tobytes()
        d["embedding"] = pickle.loads(emb_data)
    return d


def fetch_all_cluster_embeddings() -> list[dict]:
    """Return id + label + summary + embedding + article_count + created_at for all clusters (RAG)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "SELECT id, label, summary, embedding, article_count, created_at FROM clusters")
        rows = fetch_rows(cursor)
    result = []
    for d in rows:
        if d["embedding"]:
            emb_data = d["embedding"]
            if isinstance(emb_data, memoryview):
                emb_data = emb_data.tobytes()
            d["embedding"] = pickle.loads(emb_data)
        result.append(d)
    return result


def cluster_count() -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "SELECT COUNT(*) FROM clusters")
        return cursor.fetchone()[0]


def clear_clusters() -> None:
    """Wipe clusters table — called before a full re-cluster run."""
    with get_connection() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "DELETE FROM clusters")
    logger.info("Clusters table cleared for re-clustering.")
    
    