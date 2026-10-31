"""RNMF-L1: robust NMF with a signed sparse error matrix (Zhang et al., 2011).

Objective, with samples stored as columns of V (shape (d, n)):

    min_{W >= 0, H >= 0, S}  ||V - W H - S||_F^2 + lam * ||S||_1

S is a signed sparse corruption estimate and is not constrained to be
nonnegative. W @ H is the recovered clean component; S must not be added to
the reconstruction when computing RRE.
"""

from numbers import Integral, Real
import time

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _real(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _factor(init, shape, name):
    factor = np.array(init, dtype=np.float64, copy=True)
    if factor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {factor.shape}")
    if not np.isfinite(factor).all() or np.any(factor < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return factor


def soft_threshold(X, threshold):
    """Return the element-wise soft thresholding sign(X) * max(|X| - t, 0).

    This is the proximal operator of t * ||.||_1 and is the exact minimiser of
    ||R - S||_F^2 + lam * ||S||_1 over S when X = R and threshold = lam / 2.
    """
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0.0)


def rnmf_l1_objective(V, W, H, S, lam):
    """Return ||V - W H - S||_F^2 + lam * ||S||_1."""
    residual = V - W @ H - S
    return float(np.sum(residual * residual) + lam * np.abs(S).sum())


def fit_rnmf_l1(V, rank, lam, max_iter=2000, tol=1e-5, seed=0, eps=1e-10,
                W_init=None, H_init=None):
    """Fit RNMF-L1 and return ``(W, H, S, history)``.

    V is a finite, nonnegative real matrix of shape (d, n), one sample per
    column (pixels normalised to [0, 1] in this project, so lam is on that
    scale). rank is the factorisation rank k. lam >= 0 weights the L1 penalty
    on S; entries of the residual V - W H smaller than lam / 2 in magnitude are
    attributed to the low-rank part, larger ones are absorbed by S. A very
    large lam forces S = 0 and reduces the method to Frobenius NMF.

    Each iteration performs, in order:

        S <- soft_threshold(V - W H, lam / 2)
        X <- V - S                                  (may contain negatives)
        W <- W * max(X H^T, 0) / (W H H^T + eps)
        H <- H * max(W^T X, 0) / (W^T W H + eps)

    The positive part in the numerators follows Zhang et al. (2011). For the
    exact S-update, X = V - S is non-negative; the positive part is a
    safeguard that keeps W and H non-negative for any X. max(., 0) is the
    exact minimiser of the Lee-Seung auxiliary function over the nonnegative
    orthant, so each step still does not increase the objective. When X >= 0
    the updates equal Lee-Seung updates.

    Initialisation: W and H are drawn from default_rng(seed) as uniform [0, 1)
    values scaled by sqrt(mean(V) / rank), unless W_init / H_init are given
    (useful for matching the baseline NMF initialisation); S starts at zero.

    Stopping rule (the same as nmf_baseline): from iteration 200 (the
    built-in min_iter) on, the fit stops when the cumulative relative
    objective change over the last 5 iterations, sum |f_t - f_{t-1}| /
    |f_{t-5}| for the 5 most recent steps, falls below tol; "converged" is
    then True. Otherwise it stops after max_iter iterations. Because that
    change is never negative, tol = 0 disables the rule and gives a fixed
    budget of exactly max_iter iterations (the setting used in main.py).

    Returns new arrays W (d, k), H (k, n), S (d, n) and a history dict with
    keys "objective" (value after every iteration), "n_iter", "converged" and
    "runtime" (seconds). V, W_init and H_init are never modified. Invalid
    types raise TypeError; invalid shapes or values raise ValueError; a
    non-finite objective during fitting raises FloatingPointError.
    """
    V = np.asarray(V)
    if V.ndim != 2:
        raise ValueError("V must be a 2D matrix with samples as columns")
    if V.dtype.kind not in "iuf":
        raise TypeError("V must contain real numeric values")
    V = V.astype(np.float64, copy=False)
    if not np.isfinite(V).all():
        raise ValueError("V must contain only finite values")
    if np.any(V < 0):
        raise ValueError("V must be nonnegative")
    d, n = V.shape
    rank = _integer(rank, "rank", 1)
    lam = _real(lam, "lam", 0.0)
    max_iter = _integer(max_iter, "max_iter", 1)
    tol = _real(tol, "tol", 0.0)
    seed = _integer(seed, "seed", 0)
    eps = _real(eps, "eps", 0.0)

    rng = np.random.default_rng(seed)
    scale = np.sqrt(V.mean() / rank) if V.size else 1.0
    if W_init is None:
        W = rng.random((d, rank)) * scale
    else:
        W = _factor(W_init, (d, rank), "W_init")
    if H_init is None:
        H = rng.random((rank, n)) * scale
    else:
        H = _factor(H_init, (rank, n), "H_init")

    objective = []
    converged = False

    window = 5
    min_iter = 200

    start = time.perf_counter()
    WH = W @ H
    for _ in range(max_iter):
        S = soft_threshold(V - WH, lam / 2.0)
        X = V - S

        HHt = H @ H.T
        W *= np.maximum(X @ H.T, 0.0) / (W @ HHt + eps)
        WtW = W.T @ W
        H *= np.maximum(W.T @ X, 0.0) / (WtW @ H + eps)

        WH = W @ H
        residual = X - WH
        value = float(
            np.sum(residual * residual)
            + lam * np.abs(S).sum()
        )

        if not np.isfinite(value):
            raise FloatingPointError(
                f"RNMF-L1 objective became non-finite at iteration "
                f"{len(objective) + 1}"
            )

        objective.append(value)

        if (len(objective) >= min_iter and len(objective) >= window + 1):
            recent_objectives = objective[-(window + 1):]

            cumulative_change = (
                    np.sum(np.abs(np.diff(recent_objectives)))
                    / (abs(recent_objectives[0]) + eps)
            )

            if cumulative_change < tol:
                converged = True
                break

    history = {
        "objective": objective,
        "n_iter": len(objective),
        "converged": converged,
        "runtime": time.perf_counter() - start,
    }
    return W, H, S, history
