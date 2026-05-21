"""Face embedding clustering helpers."""

from __future__ import annotations

import inspect
from typing import Dict, Iterable, List

import numpy as np

from .recognition import cosine_similarity, normalize_embedding


def cluster_embeddings(
    embeddings: Iterable[np.ndarray],
    threshold: float = 0.72,
    min_samples: int = 2,
) -> List[int]:
    labels, _ = cluster_embeddings_dbscan(
        embeddings,
        distance_threshold=max(0.01, 1.0 - float(threshold)),
        min_samples=min_samples,
    )
    return labels


def cluster_embeddings_dbscan(
    embeddings: Iterable[np.ndarray],
    distance_threshold: float = 0.28,
    min_samples: int = 2,
) -> tuple[List[int], str]:
    normalized = [normalize_embedding(embedding) for embedding in embeddings]
    vectors = [embedding for embedding in normalized if embedding is not None]
    if not vectors:
        return [], "none"
    matrix = np.vstack(vectors)
    eps = max(0.01, float(distance_threshold))
    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=eps, min_samples=max(1, int(min_samples)), metric="cosine").fit_predict(matrix)
        return [int(label) for label in labels], "DBSCAN"
    except Exception:
        labels: List[int] = []
        centroids: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        next_label = 0
        for vector in matrix:
            best_label = None
            best_score = -1.0
            for label, centroid in centroids.items():
                score = cosine_similarity(vector, centroid)
                if score > best_score:
                    best_label = label
                    best_score = score
            if best_label is not None and (1.0 - best_score) <= eps:
                labels.append(best_label)
                counts[best_label] += 1
                centroids[best_label] = normalize_embedding(
                    centroids[best_label] * (counts[best_label] - 1) + vector
                )
            else:
                labels.append(next_label)
                centroids[next_label] = vector
                counts[next_label] = 1
                next_label += 1
        return labels, "centroid fallback"


def cluster_embeddings_hdbscan_auto(
    embeddings: Iterable[np.ndarray],
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> tuple[List[int], str]:
    """Cluster normalized embeddings with HDBSCAN and automatic density thresholds.

    HDBSCAN chooses cluster density thresholds internally. If the optional
    implementation is unavailable, fall back to an automatic-epsilon DBSCAN so
    the album workflow remains usable without adding a hard GUI dependency.
    """

    normalized = [normalize_embedding(embedding) for embedding in embeddings]
    vectors = [embedding for embedding in normalized if embedding is not None]
    if not vectors:
        return [], "none"
    if len(vectors) == 1:
        return [0], "HDBSCAN"
    matrix = np.vstack(vectors).astype(np.float32)
    distance_matrix = _cosine_distance_matrix(matrix)
    cluster_size = max(2, int(min_cluster_size))
    samples = max(1, int(min_samples if min_samples is not None else cluster_size))
    try:
        from sklearn.cluster import HDBSCAN

        kwargs = {
            "min_cluster_size": cluster_size,
            "min_samples": samples,
            "metric": "precomputed",
        }
        if "copy" in inspect.signature(HDBSCAN).parameters:
            kwargs["copy"] = True
        labels = HDBSCAN(**kwargs).fit_predict(distance_matrix)
        return [int(label) for label in labels], "HDBSCAN"
    except Exception:
        try:
            import hdbscan

            labels = hdbscan.HDBSCAN(
                min_cluster_size=cluster_size,
                min_samples=samples,
                metric="precomputed",
            ).fit_predict(distance_matrix)
            return [int(label) for label in labels], "HDBSCAN"
        except Exception:
            eps = _auto_dbscan_eps(distance_matrix, samples)
            labels, _ = cluster_embeddings_dbscan(matrix, distance_threshold=eps, min_samples=samples)
            return labels, "HDBSCAN unavailable; auto DBSCAN fallback"


def _cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    distances = (1.0 - similarity).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    return distances


def _auto_dbscan_eps(distance_matrix: np.ndarray, min_samples: int) -> float:
    if distance_matrix.shape[0] <= 1:
        return 0.28
    sorted_distances = np.sort(distance_matrix, axis=1)
    neighbor_index = min(max(1, int(min_samples) - 1), sorted_distances.shape[1] - 1)
    neighbor_distances = sorted_distances[:, neighbor_index]
    eps = float(np.percentile(neighbor_distances, 75))
    return min(0.45, max(0.12, eps))
