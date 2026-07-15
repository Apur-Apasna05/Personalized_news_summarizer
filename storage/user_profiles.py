"""
storage/user_profiles.py
Stores and updates per-user cluster weight vectors in SQLite.

Schema
------
user_profiles
  user_id       TEXT  PRIMARY KEY
  weights       TEXT  JSON dict  {cluster_id: float}
  created_at    TEXT
  updated_at    TEXT

user_feedback_log
  id            INTEGER PK
  user_id       TEXT
  cluster_id    INTEGER
  signal        TEXT    ('thumbs_up' | 'thumbs_down' | 'dwell')
  value         REAL    (1.0 / -1.0 / dwell_seconds)
  created_at    TEXT

Design notes
------------
- Weights are stored as a JSON dict so we never need to migrate columns
  when new clusters appear.
- Weight vector is sparse: only clusters the user has interacted with
  appear. Unknown clusters get a default weight of 0.0 at retrieval time.
- The feedback log is append-only for auditability; the weights table
  holds the current rolled-up state.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
import urllib.parse

from config.settings import DB_PATH, DATABASE_URL

logger = logging.getLogger(__name__)

IS_POSTGRES = bool(DATABASE_URL)

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


# ── Connection (reuses the same DB file as articles/clusters) ─────────────────

@contextmanager
def _conn():
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
        import os
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
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

def init_user_tables() -> None:
    """Create user profile tables if they don't exist."""
    with _conn() as conn:
        cursor = conn.cursor()
        if IS_POSTGRES:
            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id    TEXT PRIMARY KEY,
                    weights    TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS user_feedback_log (
                    id         SERIAL PRIMARY KEY,
                    user_id    TEXT    NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    signal     TEXT    NOT NULL,
                    value      REAL    NOT NULL,
                    created_at TEXT    NOT NULL
                )
            """)
        else:
            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id    TEXT PRIMARY KEY,
                    weights    TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            db_execute(cursor, """
                CREATE TABLE IF NOT EXISTS user_feedback_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT    NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    signal     TEXT    NOT NULL,
                    value      REAL    NOT NULL,
                    created_at TEXT    NOT NULL
                )
            """)
        db_execute(cursor, 
            "CREATE INDEX IF NOT EXISTS idx_feedback_user "
            "ON user_feedback_log(user_id)"
        )
    logger.info("User profile tables ready (Postgres: %s).", IS_POSTGRES)


# ── Profile CRUD ──────────────────────────────────────────────────────────────

def get_or_create_profile(user_id: str) -> dict:
    """
    Return the user's profile dict, creating it with empty weights if new.

    Returns:
        {user_id, weights: {cluster_id: float}, created_at, updated_at}
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cursor = conn.cursor()
        if IS_POSTGRES:
            db_execute(cursor,
                """
                INSERT INTO user_profiles (user_id, weights, created_at, updated_at)
                VALUES (?, '{}', ?, ?)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (user_id, now, now),
            )
        else:
            db_execute(cursor,
                """
                INSERT OR IGNORE INTO user_profiles (user_id, weights, created_at, updated_at)
                VALUES (?, '{}', ?, ?)
                """,
                (user_id, now, now),
            )
        
        db_execute(cursor, "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = fetch_row(cursor)
    
    d = dict(row)
    if isinstance(d["weights"], str):
        d["weights"] = json.loads(d["weights"])
    # Convert str keys back to int (JSON serialises dict keys as strings)
    d["weights"] = {int(k): v for k, v in d["weights"].items()}
    return d


def save_weights(user_id: str, weights: dict) -> None:
    """Persist an updated weight dict for the user."""
    now = datetime.now(timezone.utc).isoformat()
    # Store int keys as strings (JSON requirement)
    serialised = json.dumps({str(k): v for k, v in weights.items()})
    with _conn() as conn:
        cursor = conn.cursor()
        db_execute(cursor,
            """
            UPDATE user_profiles
            SET weights = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (serialised, now, user_id),
        )


def log_feedback(
    user_id: str, cluster_id: int, signal: str, value: float
) -> None:
    """Append one feedback event to the audit log."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cursor = conn.cursor()
        db_execute(cursor,
            """
            INSERT INTO user_feedback_log
                (user_id, cluster_id, signal, value, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, cluster_id, signal, value, now),
        )


def get_feedback_history(user_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent feedback events for a user."""
    with _conn() as conn:
        cursor = conn.cursor()
        db_execute(cursor,
            """
            SELECT * FROM user_feedback_log
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = fetch_rows(cursor)
    return rows


def delete_profile(user_id: str) -> None:
    """Remove a user profile and their feedback log (GDPR helper)."""
    with _conn() as conn:
        cursor = conn.cursor()
        db_execute(cursor, "DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        db_execute(cursor, "DELETE FROM user_feedback_log WHERE user_id = ?", (user_id,))
    logger.info("Deleted profile + feedback log for user '%s'.", user_id)