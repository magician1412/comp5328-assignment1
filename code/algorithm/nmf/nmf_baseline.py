"""Frobenius-norm NMF with Lee-Seung multiplicative updates (Lee & Seung, 2001).

Samples are stored as columns of X (shape (d, n)). Only NumPy is used.
"""

import numpy as np

def nmf_baseline(
    X,
    rank,
    max_iter=500,
    tol=1e-5,
    random_state=42,
    W_init=None,
    H_init=None
):
    """Fit X ≈ D @ R by minimising ||X - D R||_F^2 subject to D, R >= 0.

    X is a nonnegative matrix of shape (d, n), one sample per column; rank is
    the factorisation rank k. D (d, k) and R (k, n) start from W_init and
    H_init when given (copied, never modified), otherwise from uniform [0, 1)
    values drawn with default_rng(random_state). Each iteration applies the
    multiplicative updates

        R <- R * (D^T X) / (D^T D R + eps)
        D <- D * (X R^T) / (D R R^T + eps),    eps = 1e-10,

    and records the objective ||X - D R||_F^2.

    Stopping rule: from iteration 200 (the built-in min_iter) on, the fit
    stops when the cumulative relative objective change over the last 5
    iterations, sum |f_t - f_{t-1}| / |f_{t-5}| for the 5 most recent steps,
    falls below tol; otherwise it stops after max_iter iterations. tol = 0
    therefore gives a fixed budget of max_iter iterations.

    Returns (D, R, objective_history, converged): the factors, the list of
    objective values (one per iteration) and whether the tolerance rule
    fired. Raises ValueError for a non-2D or negative X, or for initial
    factors of the wrong shape or with negative entries.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    if np.any(X < 0):
        raise ValueError("NMF requires non-negative input.")

    n_features, n_samples = X.shape

    rng = np.random.default_rng(random_state)

    # Initialization
    if W_init is None:
        D = rng.random((n_features, rank))
    else:
        D = np.array(W_init, dtype=float, copy=True)

    if H_init is None:
        R = rng.random((rank, n_samples))
    else:
        R = np.array(H_init, dtype=float, copy=True)

    if D.shape != (n_features, rank):
        raise ValueError("W_init has incorrect shape.")

    if R.shape != (rank, n_samples):
        raise ValueError("H_init has incorrect shape.")

    if np.any(D < 0) or np.any(R < 0):
        raise ValueError("Initial factors must be non-negative.")

    eps = 1e-10

    objective_history = []
    min_iter = 200
    converged = False
    window = 5

    for _ in range(max_iter):

        # Update R
        R *= (D.T @ X) / (D.T @ D @ R + eps)

        # Update D using the updated R
        D *= (X @ R.T) / (D @ R @ R.T + eps)

        # Objective: ||X - DR||_F^2
        objective = np.linalg.norm(
            X - D @ R,
            ord="fro"
        ) ** 2

        objective_history.append(objective)

        # Convergence check
        # do not stop before min_iter iterations.
        # After that, check the cumulative relative objective change
        # over the last 5 iterations.
        if (
                len(objective_history) >= min_iter
                and len(objective_history) >= window + 1
        ):
            recent_objectives = objective_history[-(window + 1):]

            cumulative_change = (
                    np.sum(np.abs(np.diff(recent_objectives)))
                    / (abs(recent_objectives[0]) + eps)
            )

            if cumulative_change < tol:
                converged = True
                break

    return D, R, objective_history, converged