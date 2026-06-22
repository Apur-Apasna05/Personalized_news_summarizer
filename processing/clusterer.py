"""
processing/clusterer.py
Groups article embeddings into topic clusters using HDBSCAN.

Why HDBSCAN over K-Means?
  - No need to specify k (number of clusters) upfront.
  - Naturally handles noise — articles that don't fit any cluster get
    label -1 and are stored as singletons rather than forced into a cluster.
  - Works well with cosine-similar embedding spaces.

UMAP pre-reduction (optional but recommended for large batches)
  - HDBSCAN's complexity scales with dimensionality.
  - Reducing 384-dim MiniLM embeddings → 10-20 dims with UMAP before
    clustering significantly speeds things up with minimal quality loss.
  - We only apply UMAP if we have enough articles (>= 50).

Output
  A list of ClusterResult objects, each holding:
    - cluster_id    : int (HDBSCAN label, -1 = noise/singleton)
    - article_indices: list of positions in the original articles list
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import hdbscan

from config.settings import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES

logger = logging.getLogger(__name__)

UMAP_THRESHOLD = 50      # apply UMAP only if >= this many articles
UMAP_N_COMPONENTS = 15   # target dimensionality after reduction


@dataclass
class ClusterResult:
    cluster_id: int
    article_indices: list[int] = field(default_factory=list)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == -1

    def __len__(self) -> int:
        return len(self.article_indices)


def _reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """
    Optionally reduce embedding dimensions with UMAP before clustering.
    Falls back to raw embeddings if umap-learn isn't installed.
    """
    if len(embeddings) < UMAP_THRESHOLD:
        return embeddings

    try:
        import umap
        logger.info(
            "Applying UMAP: %s → (%d, %d)",
            embeddings.shape, len(embeddings), UMAP_N_COMPONENTS,
        )
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.0,        # tight clusters → better HDBSCAN input
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
        logger.info("UMAP complete → shape %s", reduced.shape)
        return reduced
    except Exception as exc:
        logger.warning("umap-learn import or execution failed (%s) — skipping UMAP reduction.", exc)
        return embeddings


def cluster_embeddings(embeddings: np.ndarray) -> list[ClusterResult]:
    """
    Run HDBSCAN on embeddings and return a list of ClusterResult objects.

    Noise points (label == -1) are each returned as their own singleton
    ClusterResult with is_noise=True. The caller decides whether to
    summarise them individually or discard them.

    Args:
        embeddings: 2D numpy array, shape (n_articles, embedding_dim).

    Returns:
        List of ClusterResult, one per unique cluster label (+ singletons).
    """
    n = len(embeddings)
    if n == 0:
        logger.warning("No embeddings to cluster.")
        return []

    if n < HDBSCAN_MIN_CLUSTER_SIZE:
        logger.warning(
            "Only %d articles — below min_cluster_size=%d. "
            "Treating all as one cluster.",
            n, HDBSCAN_MIN_CLUSTER_SIZE,
        )
        return [ClusterResult(cluster_id=0, article_indices=list(range(n)))]

    # Dimensionality reduction
    reduced = _reduce_dimensions(embeddings)

    # HDBSCAN
    logger.info("Running HDBSCAN on %d articles...", n)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",          # euclidean on UMAP-reduced space
        cluster_selection_method="eom",  # excess of mass — stable clusters
    )
    labels = clusterer.fit_predict(reduced)

    # Group indices by label
    cluster_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(idx)

    results = []
    noise_indices = cluster_map.pop(-1, [])

    # Real clusters
    for cid, indices in sorted(cluster_map.items()):
        results.append(ClusterResult(cluster_id=cid, article_indices=indices))

    # Noise articles → individual singletons (each gets its own summary)
    for idx in noise_indices:
        results.append(ClusterResult(cluster_id=-1, article_indices=[idx]))

    n_clusters = len(cluster_map)
    n_noise    = len(noise_indices)
    logger.info(
        "HDBSCAN found %d clusters, %d noise/singleton articles",
        n_clusters, n_noise,
    )
    return results