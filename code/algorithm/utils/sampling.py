"""Deterministic global sampling of columns, independent of image loading."""

from math import floor
from numbers import Integral, Real

import numpy as np


def sample_columns(V_clean, labels, fraction=0.9, seed=42):
    """Return ``(V_sub, labels_sub, indices)`` without modifying clean inputs.

    V_clean must have shape (d, n) and labels must have shape (n,). Inputs are
    converted with np.asarray; their dtypes and selected values are preserved.
    Labels are used only for aligned slicing: sampling is uniform across ALL n
    columns, without replacement or class stratification. Classes may disappear
    from a subset; labels are never renumbered.

    Select floor(fraction * n) columns, with finite real fraction in (0, 1] and
    at least one selected column. The seed must be a non-negative Python/NumPy
    integer (not a boolean). None and mutable RNG objects are not accepted.
    A fresh np.random.default_rng(seed) makes membership reproducible without
    reading or modifying the global NumPy RNG state.

    Returned indices are a 1D integer array sorted in original-dataset order.
    V_sub equals V_clean[:, indices]; labels_sub equals labels[indices]. NumPy
    advanced indexing creates independent array storage, including fraction=1,
    so later assignments to outputs do not modify the inputs. As with NumPy
    generally, object-dtype arrays do not deep-copy the referenced Python objects.

    Raises TypeError for nonnumeric fractions or unsupported seed types, and
    ValueError for invalid dimensions, lengths, fraction ranges, empty subsets,
    or negative seeds.
    """
    V_clean = np.asarray(V_clean)
    labels = np.asarray(labels)
    if V_clean.ndim != 2:
        raise ValueError("V_clean must be a 2D matrix with samples as columns")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    n = V_clean.shape[1]
    if labels.shape[0] != n:
        raise ValueError("labels length must match the number of V_clean columns")

    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real numeric scalar")
    # This comparison also rejects NaN and both infinities before conversion.
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and satisfy 0 < fraction <= 1")
    n_subset = floor(float(fraction) * n)
    if n_subset < 1:
        raise ValueError("floor(fraction * n) must select at least one column")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    if n_subset == n:
        indices = np.arange(n, dtype=np.intp)
    else:
        indices = np.sort(rng.choice(n, size=n_subset, replace=False))
    return V_clean[:, indices], labels[indices], indices
