"""Evaluation metrics for column-oriented NMF factors.

Scikit-learn is imported only inside the clustering function.  The NMF
implementations themselves therefore remain restricted to NumPy/SciPy as
required by the assignment.
"""

from numbers import Integral

import numpy as np


def _finite_matrix(value, name):
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite real values")
    return matrix.astype(np.float64, copy=False)


def relative_reconstruction_error(V_clean, W, H):
    """Return ``||V_clean - W @ H||_F / ||V_clean||_F``.

    ``V_clean`` must contain the clean columns corresponding exactly to the
    contaminated columns used for fitting.  For RNMF-L1, pass the recovered
    low-rank factors W and H only; the sparse corruption matrix S is excluded.
    """
    V_clean = _finite_matrix(V_clean, "V_clean")
    W = _finite_matrix(W, "W")
    H = _finite_matrix(H, "H")
    if W.shape[0] != V_clean.shape[0]:
        raise ValueError("W rows must match V_clean rows")
    if H.shape[1] != V_clean.shape[1]:
        raise ValueError("H columns must match V_clean columns")
    if W.shape[1] != H.shape[0]:
        raise ValueError("W columns must match H rows")
    denominator = np.linalg.norm(V_clean, ord="fro")
    if denominator == 0.0:
        raise ValueError("RRE is undefined when V_clean has zero Frobenius norm")
    return float(np.linalg.norm(V_clean - W @ H, ord="fro") / denominator)


def masked_relative_reconstruction_error(V_clean, W, H, mask):
    """Return ``||(V_clean - W @ H)[mask]||_F / ||V_clean[mask]||_F``.

    ``mask`` is a boolean matrix with the shape of ``V_clean``. Pass the
    occlusion mask for the error on occluded pixels and ``~mask`` for the
    error on unoccluded pixels. Returns NaN when ``mask`` selects no entries
    (for example the occluded part when m = 0) or when the selected clean
    entries are all zero.
    """
    V_clean = _finite_matrix(V_clean, "V_clean")
    W = _finite_matrix(W, "W")
    H = _finite_matrix(H, "H")
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != V_clean.shape:
        raise ValueError("mask must be a boolean matrix with the shape of V_clean")
    if W.shape[0] != V_clean.shape[0] or H.shape[1] != V_clean.shape[1] \
            or W.shape[1] != H.shape[0]:
        raise ValueError("W @ H must have the shape of V_clean")
    denominator = np.linalg.norm(V_clean[mask])
    if not mask.any() or denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((V_clean - W @ H)[mask]) / denominator)


def clustering_scores(H, labels, seed=0, n_init=10, n_clusters=None):
    """Cluster ``H.T`` and return accuracy, NMI, and both label vectors.

    Accuracy follows the tutorial's majority-class mapping for each cluster.
    NMI is computed from the ground-truth labels and raw cluster IDs because it
    is permutation invariant.  KMeans is used only for evaluation.
    """
    H = _finite_matrix(H, "H")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != H.shape[1]:
        raise ValueError("labels must be 1D and match the number of H columns")
    if labels.size == 0:
        raise ValueError("at least one sample is required for clustering")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_init, (bool, np.bool_)) or not isinstance(n_init, Integral):
        raise TypeError("n_init must be a positive integer")
    if n_init < 1:
        raise ValueError("n_init must be a positive integer")

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    classes = np.unique(labels)
    if n_clusters is None:
        n_clusters = classes.size
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(
        n_clusters, Integral
    ):
        raise TypeError("n_clusters must be a positive integer")
    if n_clusters < 1 or n_clusters > labels.size:
        raise ValueError("n_clusters must be between 1 and the sample count")
    raw = KMeans(
        n_clusters=int(n_clusters), random_state=int(seed),
        n_init=int(n_init),
    ).fit_predict(H.T)
    mapped = np.empty(labels.shape, dtype=labels.dtype)
    for cluster_id in np.unique(raw):
        members = labels[raw == cluster_id]
        values, counts = np.unique(members, return_counts=True)
        mapped[raw == cluster_id] = values[np.argmax(counts)]
    return {
        "accuracy": float(np.mean(mapped == labels)),
        "nmi": float(normalized_mutual_info_score(labels, raw)),
        "cluster_ids": raw,
        "predicted_labels": mapped,
    }
