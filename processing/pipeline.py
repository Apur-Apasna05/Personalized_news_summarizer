"""
processing/pipeline.py
Phase 2 orchestrator — the "slow loop".

Steps
-----
1. Fetch unprocessed articles from SQLite (Phase 1 output).
2. Embed each article (title + body snippet) with MiniLM.
3. Cluster embeddings with HDBSCAN.
4. For each cluster: call Ollama to get a label + summary.
5. Embed the summary (this vector goes into the vector DB in Phase 3).
6. Persist each cluster to the clusters table.
7. Mark all processed articles as is_processed = 1.

Re-clustering strategy
  On each slow-loop run we only process NEW (unprocessed) articles.
  We do NOT re-cluster the entire history every run — that would be
  expensive and unstable. Full re-clustering is triggered manually
  or by a weekly scheduler job (Phase 3+).
"""

import logging

import numpy as np

from config.settings import PROCESSING_MIN_BATCH
from storage.database import (
    fetch_unprocessed,
    mark_processed,
    upsert_cluster,
    cluster_count,
    article_count,
)
from processing.embedder  import embed_texts, embed_single, article_to_text
from processing.clusterer import cluster_embeddings
from processing.summarizer import summarize_cluster

logger = logging.getLogger(__name__)


def run_processing_pipeline(force: bool = False) -> dict:
    """
    Run one full Phase 2 cycle.

    Args:
        force: If True, run even if the batch is smaller than
               PROCESSING_MIN_BATCH (useful for testing).

    Returns:
        Summary dict with counts.
    """
    logger.info("=" * 50)
    logger.info("Processing pipeline starting")
    logger.info("=" * 50)

    # ── 1. Fetch unprocessed articles ─────────────────────────
    articles = fetch_unprocessed(limit=500)
    n = len(articles)
    logger.info("Unprocessed articles: %d", n)

    if n == 0:
        logger.info("Nothing to process.")
        return {"status": "skipped", "reason": "no_unprocessed_articles"}

    if n < PROCESSING_MIN_BATCH and not force:
        logger.info(
            "Batch too small (%d < %d). Use force=True to override.",
            n, PROCESSING_MIN_BATCH,
        )
        return {"status": "skipped", "reason": "batch_too_small", "count": n}

    # ── 2. Embed articles ─────────────────────────────────────
    logger.info("Embedding %d articles...", n)
    texts = [article_to_text(a) for a in articles]
    embeddings = embed_texts(texts)

    # ── 3. Cluster ────────────────────────────────────────────
    logger.info("Clustering...")
    cluster_results = cluster_embeddings(embeddings)
    logger.info("Produced %d cluster(s)", len(cluster_results))

    # ── 4–6. Summarise + embed summary + persist ──────────────
    clusters_saved  = 0
    articles_processed = []

    for cr in cluster_results:
        cluster_articles = [articles[i] for i in cr.article_indices]
        article_ids      = [a["id"] for a in cluster_articles]

        # 4. Summarise
        label, summary = summarize_cluster(cluster_articles)

        # 5. Embed summary  (stored for Phase 3 vector DB / RAG)
        summary_embedding: np.ndarray | None = None
        if summary:
            try:
                summary_embedding = embed_single(summary)
            except Exception as exc:
                logger.warning("Failed to embed summary for '%s': %s", label, exc)

        # 6. Persist
        cluster_id = upsert_cluster(
            label       = label,
            summary     = summary,
            article_ids = article_ids,
            embedding   = summary_embedding,
        )
        clusters_saved += 1
        articles_processed.extend(article_ids)

        logger.info(
            "Cluster %d saved — '%s' (%d articles)",
            cluster_id, label, len(article_ids),
        )

    # ── 7. Mark articles as processed ─────────────────────────
    mark_processed(articles_processed)
    logger.info("Marked %d articles as processed.", len(articles_processed))

    # ── Summary ───────────────────────────────────────────────
    db = article_count()
    summary_stats = {
        "status":            "ok",
        "articles_processed": len(articles_processed),
        "clusters_created":  clusters_saved,
        "total_clusters":    cluster_count(),
        "db_total":          db["total"],
        "db_unprocessed":    db["unprocessed"],
    }

    logger.info(
        "Processing complete — %d articles → %d clusters (total clusters: %d)",
        len(articles_processed),
        clusters_saved,
        summary_stats["total_clusters"],
    )
    logger.info("=" * 50)
    return summary_stats