import numpy as np

from insightface.gui.core.clustering import cluster_embeddings_dbscan, cluster_embeddings_hdbscan_auto


def test_dbscan_default_distance_threshold_groups_near_faces():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.08], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.08, 0.98], dtype=np.float32),
    ]

    labels, algorithm = cluster_embeddings_dbscan(embeddings, min_samples=2)

    assert algorithm in {"DBSCAN", "centroid fallback"}
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_hdbscan_auto_returns_labels_without_threshold_input():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.08], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.08, 0.98], dtype=np.float32),
    ]

    labels, algorithm = cluster_embeddings_hdbscan_auto(
        embeddings,
        min_cluster_size=2,
        min_samples=1,
    )

    assert "HDBSCAN" in algorithm
    assert len(labels) == 4
    assert all(isinstance(label, int) for label in labels)
