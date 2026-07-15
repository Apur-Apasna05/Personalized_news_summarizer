"""
storage/vector_store.py
ChromaDB upsert / query layer for Phase 3.

Responsibilities
----------------
- Maintain a persistent ChromaDB collection called "news_clusters".
- upsert_clusters()  : push cluster summaries (+ pre-computed embeddings)
  from SQLite into ChromaDB.  Called once after each Phase 2 run and
  after a full re-cluster.
- query_clusters()   : embed a query string and return the top-k most
  similar cluster summaries with their metadata.
- sync_from_db()     : convenience wrapper — reads ALL clusters from
  SQLite and pushes them to Chroma (full refresh).

Design notes
------------
- We use ChromaDB's built-in embedding function for NEW insertions so
  Chroma can re-embed on its own if needed, but we *also* supply the
  pre-computed numpy embeddings we already have from Phase 2 to avoid
  running the model twice.
- Document IDs in Chroma are "cluster_<sqlite_id>" for easy cross-reference.
- Chroma is opened in persistent mode so data survives process restarts.
"""

import logging
import os
from typing import Optional

import numpy as np

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)

# Check if we should use ChromaDB or fallback to DB-based search
USE_CHROMA = False
if not GEMINI_API_KEY:
    try:
        import chromadb
        USE_CHROMA = True
    except ImportError:
        logger.info("chromadb is not installed. Will use DB-based NumPy vector search.")

# Module-level client + collection cache
_client = None
_collection = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_collection():
    """Return (and lazily initialise) the ChromaDB collection."""
    global _client, _collection
    if not USE_CHROMA:
        return None

    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is not installed. Run:  pip install chromadb"
        ) from exc

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    logger.info("Opening ChromaDB at %s", CHROMA_PERSIST_DIR)

    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # get_or_create so we never raise on restart
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for retrieval
    )
    logger.info(
        "ChromaDB collection '%s' ready (%d docs)",
        CHROMA_COLLECTION,
        _collection.count(),
    )
    return _collection


def _cluster_doc_id(sqlite_id: int) -> str:
    return f"cluster_{sqlite_id}"


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_clusters(clusters: list[dict]) -> int:
    """
    Upsert a list of cluster dicts into ChromaDB (or do nothing if fallback is active).

    Each cluster dict must have at least:
        id        : int   — SQLite cluster id
        summary   : str   — LLM-generated summary
        label     : str   — short topic label
        embedding : np.ndarray | None — pre-computed summary embedding

    Clusters without a precomputed embedding are skipped (Chroma could
    re-embed them, but we rely on our own model for consistency).

    Returns the number of clusters successfully upserted.
    """
    if not clusters:
        logger.info("upsert_clusters: nothing to upsert.")
        return 0

    if not USE_CHROMA:
        logger.info("upsert_clusters (fallback): skipping ChromaDB write.")
        return sum(1 for c in clusters if c.get("embedding") is not None)

    col = _get_collection()

    ids        = []
    documents  = []
    embeddings = []
    metadatas  = []

    for c in clusters:
        emb = c.get("embedding")
        if emb is None:
            logger.debug("Skipping cluster %s — no embedding.", c.get("id"))
            continue

        # Chroma expects Python lists of floats
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()

        ids.append(_cluster_doc_id(c["id"]))
        documents.append(c.get("summary", ""))
        embeddings.append(emb)
        metadatas.append({
            "sqlite_id":    int(c["id"]),
            "label":        c.get("label", ""),
            "article_count": int(c.get("article_count", 0)),
            "created_at":   c.get("created_at", ""),
        })

    if not ids:
        logger.warning("upsert_clusters: all clusters lacked embeddings.")
        return 0

    col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Upserted %d clusters into ChromaDB.", len(ids))
    return len(ids)


def query_clusters(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Embed `query` and return the top-k most relevant cluster summaries.

    Args:
        query          : Natural-language question or keyword.
        top_k          : Maximum number of results to return.
        score_threshold: Minimum cosine similarity (0–1) to include a result.
                         Chroma returns *distance* (lower = better for cosine),
                         so we convert: similarity = 1 - distance.

    Returns:
        List of dicts, each with keys:
            id, label, summary, article_count, created_at,
            similarity (float 0-1)
        Sorted by similarity descending.
    """
    if not USE_CHROMA:
        from storage.database import fetch_all_cluster_embeddings
        from processing.embedder import embed_single

        logger.info("query_clusters (fallback): performing DB-based vector search.")
        clusters = fetch_all_cluster_embeddings()
        if not clusters:
            logger.warning("No clusters found in database.")
            return []

        # Embed the query
        query_embedding = embed_single(query)

        output = []
        for c in clusters:
            emb = c.get("embedding")
            if emb is None:
                continue

            # Calculate cosine similarity using numpy
            dot_prod = np.dot(query_embedding, emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_emb = np.linalg.norm(emb)
            similarity = float(dot_prod / (norm_q * norm_emb)) if (norm_q * norm_emb) > 0 else 0.0

            if similarity < score_threshold:
                continue

            output.append({
                "id":            c["id"],
                "label":         c.get("label", ""),
                "summary":       c.get("summary", ""),
                "article_count": c.get("article_count", 0),
                "created_at":    c.get("created_at", ""),
                "similarity":    round(similarity, 4),
            })

        # Sort descending by similarity
        output.sort(key=lambda x: x["similarity"], reverse=True)
        return output[:top_k]

    from processing.embedder import embed_single

    col = _get_collection()
    if col.count() == 0:
        logger.warning("ChromaDB collection is empty — run sync_from_db() first.")
        return []

    # Embed the query using the same model used for cluster summaries
    query_embedding = embed_single(query).tolist()

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Chroma cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity in [0, 1]
        similarity = max(0.0, 1.0 - dist / 2.0)

        if similarity < score_threshold:
            continue

        output.append({
            "id":            meta.get("sqlite_id"),
            "label":         meta.get("label", ""),
            "summary":       doc,
            "article_count": meta.get("article_count", 0),
            "created_at":    meta.get("created_at", ""),
            "similarity":    round(similarity, 4),
        })

    logger.info(
        "query_clusters('%s') → %d results (top_k=%d)", query, len(output), top_k
    )
    return output


def sync_from_db() -> int:
    """
    Full refresh: read ALL clusters (with embeddings) from SQLite/Postgres and
    upsert them into ChromaDB.

    Call this:
      - After every Phase 2 processing run.
      - After a full re-cluster (clear_clusters + reprocess).
      - On application startup to restore the Chroma state from database truth.

    Returns the number of clusters synced.
    """
    from storage.database import fetch_all_cluster_embeddings

    logger.info("sync_from_db: reading all clusters from database...")
    clusters = fetch_all_cluster_embeddings()
    logger.info("sync_from_db: found %d clusters in database.", len(clusters))

    if not clusters:
        return 0

    return upsert_clusters(clusters)


def collection_stats() -> dict:
    """Return basic stats about the ChromaDB collection (or DB counts in fallback)."""
    if not USE_CHROMA:
        from storage.database import cluster_count
        return {
            "collection": "database_fallback",
            "persist_dir": "N/A (database fallback)",
            "doc_count": cluster_count(),
        }

    col = _get_collection()
    return {
        "collection": CHROMA_COLLECTION,
        "persist_dir": CHROMA_PERSIST_DIR,
        "doc_count": col.count(),
    }


def delete_cluster(sqlite_id: int) -> None:
    """Remove a single cluster from ChromaDB by its SQLite id."""
    if not USE_CHROMA:
        return

    col = _get_collection()
    col.delete(ids=[_cluster_doc_id(sqlite_id)])
    logger.info("Deleted cluster %d from ChromaDB.", sqlite_id)


def reset_collection() -> None:
    """
    Wipe the entire ChromaDB collection and recreate it.
    Use before a full re-cluster run so stale entries don't persist.
    """
    if not USE_CHROMA:
        return

    global _collection
    col = _get_collection()
    col.delete(where={"sqlite_id": {"$gte": 0}})   # delete all docs
    _collection = None   # force re-init on next call
    logger.warning("ChromaDB collection '%s' has been wiped.", CHROMA_COLLECTION)
