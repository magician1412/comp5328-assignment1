"""Reproducible white square occlusion for normalized, column-oriented images."""

from numbers import Integral

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def apply_occlusion(V_clean, image_shape, block_size, num_blocks, seed):
    """Return ``(V_noisy, mask, positions)`` without modifying V_clean.

    V_clean is a real numeric matrix of shape (d, n), with finite values in
    [0, 1]. image_shape is (height, width), with positive integer dimensions
    whose product is d. Each column represents one C-order flattened image.
    Floating input dtypes are preserved; integer 0/1 inputs become float64.
    Boolean, complex, object and other non-real-numeric arrays are rejected.

    For each image, independently draw exactly num_blocks top-left positions
    uniformly among all positions where a block_size x block_size square fits.
    Overlap and repeated positions are allowed, without rejection or clipping.
    Covered pixels are set to exactly 1.0. mask is a new bool array of shape
    (d, n) representing the union of the squares, including covered pixels that
    were already white. positions is a new int64 array of shape
    (n, num_blocks, 2), storing (row, column) for every block in draw order.

    block_size must be a positive integer fitting both dimensions. num_blocks
    must be a non-negative integer. seed must be a non-negative Python/NumPy
    integer. Boolean integer parameters are rejected. A fresh default_rng(seed)
    provides reproducibility without using global NumPy RNG state.

    V_noisy has independent mutable storage. With num_blocks=0, it is an
    unchanged-value copy, mask is all False and positions has shape (n, 0, 2).
    All arguments, including block_size and seed, are still validated. A matrix
    with zero columns is allowed and returns correspondingly empty outputs.
    Invalid types raise TypeError; invalid shapes, ranges or values raise
    ValueError. No normalization or value clipping is performed here.
    """
    V_clean = np.asarray(V_clean)
    if V_clean.ndim != 2:
        raise ValueError("V_clean must be a 2D matrix with samples as columns")
    if V_clean.dtype.kind not in "iuf":
        raise TypeError("V_clean must contain real numeric values")
    if not np.isfinite(V_clean).all():
        raise ValueError("V_clean must contain only finite values")
    if np.any(V_clean < 0) or np.any(V_clean > 1):
        raise ValueError("V_clean values must lie in [0, 1]")

    try:
        dimensions = tuple(image_shape)
    except TypeError as exc:
        raise TypeError("image_shape must contain two integer dimensions") from exc
    if len(dimensions) != 2:
        raise ValueError("image_shape must contain exactly (height, width)")
    height = _integer(dimensions[0], "height", 1)
    width = _integer(dimensions[1], "width", 1)
    if height * width != V_clean.shape[0]:
        raise ValueError("height * width must match the number of V_clean rows")
    block_size = _integer(block_size, "block_size", 1)
    if block_size > height or block_size > width:
        raise ValueError("block_size must fit within both height and width")
    num_blocks = _integer(num_blocks, "num_blocks", 0)
    seed = _integer(seed, "seed", 0)

    n = V_clean.shape[1]
    rng = np.random.default_rng(seed)
    positions = rng.integers(
        0, (height - block_size + 1, width - block_size + 1),
        size=(n, num_blocks, 2), dtype=np.int64,
    )
    mask = np.zeros(V_clean.shape, dtype=bool)
    # C-order index p = row * width + column; image index remains the last axis.
    mask_images = mask.reshape(height, width, n, order="C")
    for j in range(n):
        for row, col in positions[j]:
            mask_images[row:row + block_size, col:col + block_size, j] = True

    dtype = V_clean.dtype if V_clean.dtype.kind == "f" else np.float64
    V_noisy = np.array(V_clean, dtype=dtype, copy=True)
    V_noisy[mask] = 1.0
    return V_noisy, mask, positions
