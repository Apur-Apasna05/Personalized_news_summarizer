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

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

# ── Connection (reuses the same DB file as articles/clusters) ─────────────────

@contextmanager
def _conn():
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


# ── Schema ────────────────────────────────────────────────────────────────────

def init_user_tables() -> None:
    """Create user profile tables if they don't exist."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id    TEXT PRIMARY KEY,
                weights    TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                cluster_id INTEGER NOT NULL,
                signal     TEXT    NOT NULL,
                value      REAL    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user "
            "ON user_feedback_log(user_id)"
        )
    logger.info("User profile tables ready.")


# ── Profile CRUD ──────────────────────────────────────────────────────────────

def get_or_create_profile(user_id: str) -> dict:
    """
    Return the user's profile dict, creating it with empty weights if new.

    Returns:
        {user_id, weights: {cluster_id: float}, created_at, updated_at}
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO user_profiles (user_id, weights, created_at, updated_at)
            VALUES (?, '{}', ?, ?)
            """,
            (user_id, now, now),
        )
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    d = dict(row)
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
        conn.execute(
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
        conn.execute(
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
        rows = conn.execute(
            """
            SELECT * FROM user_feedback_log
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_profile(user_id: str) -> None:
    """Remove a user profile and their feedback log (GDPR helper)."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        conn.execute(
            "DELETE FROM user_feedback_log WHERE user_id = ?", (user_id,)
        )
    logger.info("Deleted profile + feedback log for user '%s'.", user_id)