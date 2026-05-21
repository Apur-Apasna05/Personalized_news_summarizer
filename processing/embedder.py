"""
processing/embedder.py
Thin wrapper around sentence-transformers.

Why a wrapper?
  - Loads the model once and caches it (expensive to reload per call).
  - Gives a clean interface: pass a list of strings, get a numpy array back.
  - Easy to swap the model later by changing EMBEDDING_MODEL in settings.py.

Input text strategy
  - For each article we embed:  title + ". " + body[:500]
    Using title + truncated body balances semantic signal vs noise.
    Full body often drifts into ads/boilerplate that hurts clustering.
"""

import logging
import numpy as np

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level cache — model is loaded once on first call
_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings.
    Returns a 2D numpy array of shape (len(texts), embedding_dim).

    Args:
        texts:      List of strings to embed.
        batch_size: Batch size for encoding. Tune down on low-RAM machines.
    """
    if not texts:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # unit vectors → cosine similarity = dot product
        convert_to_numpy=True,
    )
    logger.info("Embedded %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Convenience wrapper for embedding one string (e.g. a cluster summary)."""
    return embed_texts([text])[0]


def article_to_text(article: dict) -> str:
    """
    Combine title + body into a single string for embedding.
    Truncates body to 500 chars to keep focus on the core topic.
    """
    title = article.get("title", "").strip()
    body  = article.get("body",  "").strip()[:500]
    return f"{title}. {body}" if body else title