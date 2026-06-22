"""
personalization/retriever.py
Personalised cluster retrieval — combines semantic similarity with
user preference weights.

Scoring formula
---------------
  final_score = (1 - PERSONALIZATION_ALPHA) * similarity
              + PERSONALIZATION_ALPHA       * user_weight

Where:
  similarity    — cosine similarity from ChromaDB (0–1)
  user_weight   — user's weight for this cluster (0–1, normalised)
  PERSONALIZATION_ALPHA — blend factor from settings (default 0.3)

At alpha=0.0  → pure semantic search (same as Phase 3)
At alpha=1.0  → pure preference-based (ignores relevance)
Default 0.3   → 70% semantic, 30% personal preference

Cold-start handling
-------------------
If the user has no weights yet (new user), we fall back to pure
semantic search (alpha effectively becomes 0).
"""

import logging
from config.settings import PERSONALIZATION_ALPHA, RAG_TOP_K
from storage.vector_store import query_clusters
from storage.user_profiles import get_or_create_profile

logger = logging.getLogger(__name__)


def personalised_retrieve(
    query: str,
    user_id: str,
    top_k: int = RAG_TOP_K,
    fetch_k: int = None,
) -> list[dict]:
    """
    Retrieve and re-rank clusters using both semantic similarity and
    user preference weights.

    Args:
        query   : Natural-language search query.
        user_id : User whose preference weights to apply.
        top_k   : Number of results to return after re-ranking.
        fetch_k : How many candidates to fetch from Chroma before
                  re-ranking. Defaults to max(top_k * 3, 20) so we
                  have enough candidates to meaningfully re-rank.

    Returns:
        List of cluster dicts (same schema as query_clusters) with an
        additional 'personalised_score' key, sorted descending.
    """
    if fetch_k is None:
        fetch_k = max(top_k * 3, 20)

    # ── 1. Semantic retrieval ─────────────────────────────────
    candidates = query_clusters(query, top_k=fetch_k, score_threshold=0.0)

    if not candidates:
        logger.info("personalised_retrieve: no candidates from vector store.")
        return []

    # ── 2. Load user weights ──────────────────────────────────
    profile = get_or_create_profile(user_id)
    weights = profile["weights"]   # {cluster_id: float}

    # If user has no feedback yet → pure semantic search
    if not weights:
        logger.debug(
            "User '%s' has no weights yet — returning pure similarity.", user_id
        )
        for c in candidates:
            c["personalised_score"] = c["similarity"]
        from rag.chain import diversify_results
        return diversify_results(candidates, top_k)

    # Max weight for normalisation (so personal scores stay in [0,1])
    max_weight = max(weights.values()) if weights else 1.0
    if max_weight == 0:
        max_weight = 1.0

    # ── 3. Re-rank ────────────────────────────────────────────
    for c in candidates:
        cid = c.get("id")
        raw_weight = weights.get(cid, 0.0)
        # Normalise weight to [0, 1]
        norm_weight = max(0.0, raw_weight / max_weight)

        c["user_weight"] = round(norm_weight, 4)
        c["personalised_score"] = round(
            (1 - PERSONALIZATION_ALPHA) * c["similarity"]
            + PERSONALIZATION_ALPHA * norm_weight,
            4,
        )

    candidates.sort(key=lambda x: x["personalised_score"], reverse=True)

    logger.info(
        "personalised_retrieve('%s', user='%s') → %d results",
        query, user_id, len(candidates[:top_k]),
    )
    from rag.chain import diversify_results
    return diversify_results(candidates, top_k)