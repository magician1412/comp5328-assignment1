"""Member 2 RNMF-L1 tests; synthetic data only, requires only NumPy.

Check the soft-thresholding S update, a non-increasing objective,
non-negative W and H, reproducibility under a fixed seed, and that sparse
corruption is absorbed by S (Section 3.3 of the report).

Run from the repository root (any directory works):
    pytest -q code/algorithm/tests/test_rnmf_l1.py

No supplied datasets are needed.
"""

import unittest

import numpy as np

from nmf.nmf_rnmf_l1 import fit_rnmf_l1, rnmf_l1_objective, soft_threshold


def _low_rank(d, n, k, seed):
    """Nonnegative rank-k matrix scaled into [0, 0.6]."""
    rng = np.random.default_rng(seed)
    V = rng.random((d, k)) @ rng.random((k, n))
    return 0.6 * V / V.max()


def _corrupt(V, fraction, seed):
    """Set a random fraction of entries to 1.0 (white), returning a new matrix."""
    rng = np.random.default_rng(seed)
    mask = rng.random(V.shape) < fraction
    noisy = V.copy()
    noisy[mask] = 1.0
    return noisy, mask


def _rre(V_clean, W, H):
    return np.linalg.norm(V_clean - W @ H) / np.linalg.norm(V_clean)


class SoftThresholdTests(unittest.TestCase):
    def test_values_and_sign(self):
        X = np.array([[-2.0, -0.5, 0.0], [0.3, 0.5, 3.0]])
        expected = np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 2.5]])
        np.testing.assert_allclose(soft_threshold(X, 0.5), expected)

    def test_is_minimiser_of_l1_problem(self):
        # S* must beat nearby perturbations of ||R - S||^2 + lam ||S||_1.
        rng = np.random.default_rng(0)
        R = rng.normal(size=(6, 5))
        lam = 0.8

        def cost(S):
            return np.sum((R - S) ** 2) + lam * np.abs(S).sum()

        best = cost(soft_threshold(R, lam / 2))
        for _ in range(50):
            S = soft_threshold(R, lam / 2) + 0.05 * rng.normal(size=R.shape)
            self.assertLessEqual(best, cost(S) + 1e-12)


class FitTests(unittest.TestCase):
    def setUp(self):
        self.V_clean = _low_rank(60, 40, 4, seed=1)
        self.V, self.mask = _corrupt(self.V_clean, 0.05, seed=2)

    def test_shapes_nonnegativity_and_history(self):
        W, H, S, history = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=50, seed=0)
        self.assertEqual(W.shape, (60, 4))
        self.assertEqual(H.shape, (4, 40))
        self.assertEqual(S.shape, (60, 40))
        self.assertTrue(np.all(W >= 0) and np.all(H >= 0))
        self.assertTrue(np.isfinite(S).all())
        self.assertEqual(history["n_iter"], len(history["objective"]))
        self.assertLessEqual(history["n_iter"], 50)
        self.assertIsInstance(history["converged"], bool)
        self.assertGreaterEqual(history["runtime"], 0.0)

    def test_objective_is_non_increasing(self):
        _, _, _, history = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=2000,
                                       tol=0.0, seed=3)
        obj = np.array(history["objective"])
        self.assertEqual(len(obj), 2000)
        self.assertTrue(np.all(np.diff(obj) <= 1e-9 * obj[:-1]))

    def test_s_update_is_soft_threshold_of_residual(self):
        rng = np.random.default_rng(4)
        W0, H0 = rng.random((60, 4)), rng.random((4, 40))
        _, _, S, _ = fit_rnmf_l1(self.V, 4, 0.2, max_iter=1, tol=0.0, W_init=W0, H_init=H0)
        R = self.V - W0 @ H0
        np.testing.assert_allclose(S, np.sign(R) * np.maximum(np.abs(R) - 0.1, 0.0), atol=1e-12)

    def test_reported_objective_matches_returned_factors(self):
        W, H, S, history = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=30, seed=0)
        self.assertAlmostEqual(history["objective"][-1],
                               rnmf_l1_objective(self.V, W, H, S, 0.2),
                               places=8)

    def test_seed_reproducibility_and_input_unchanged(self):
        before = self.V.copy()
        a = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=20, seed=5)
        b = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=20, seed=5)
        c = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=20, seed=6)
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c[0]))
        np.testing.assert_array_equal(self.V, before)

    def test_explicit_initialisation_is_used_and_not_modified(self):
        rng = np.random.default_rng(9)
        W0, H0 = rng.random((60, 4)), rng.random((4, 40))
        W0_copy, H0_copy = W0.copy(), H0.copy()
        a = fit_rnmf_l1(self.V, 4, 0.2, max_iter=10, seed=0, W_init=W0, H_init=H0)
        b = fit_rnmf_l1(self.V, 4, 0.2, max_iter=10, seed=99, W_init=W0, H_init=H0)
        np.testing.assert_array_equal(a[0], b[0])  # seed unused when both given
        np.testing.assert_array_equal(W0, W0_copy)
        np.testing.assert_array_equal(H0, H0_copy)

    def test_huge_lambda_gives_zero_sparse_error(self):
        _, _, S, _ = fit_rnmf_l1(self.V, 4, lam=1e6, max_iter=20, seed=0)
        self.assertFalse(np.any(S))

    def test_sparse_corruption_is_absorbed_by_S(self):
        # Same data, rank and initialisation; only lam differs. lam = 1e6 is
        # plain Frobenius NMF, so this checks the robustness mechanism itself.
        W_r, H_r, S, _ = fit_rnmf_l1(self.V, 4, lam=0.2, max_iter=1000, seed=0)
        W_f, H_f, _, _ = fit_rnmf_l1(self.V, 4, lam=1e6, max_iter=1000, seed=0)
        rre_robust = _rre(self.V_clean, W_r, H_r)
        rre_frobenius = _rre(self.V_clean, W_f, H_f)
        self.assertLess(rre_robust, 0.5 * rre_frobenius)
        # Most of S's mass lies on corrupted entries and is positive there.
        mass = np.abs(S)
        self.assertGreater(mass[self.mask].sum() / mass.sum(), 0.9)
        self.assertGreater(np.mean(S[self.mask] > 0), 0.9)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.V = _low_rank(10, 8, 2, seed=0)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            fit_rnmf_l1(-self.V - 0.1, 2, 0.1)
        bad = self.V.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_rnmf_l1(bad, 2, 0.1)
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V[:, 0], 2, 0.1)
        with self.assertRaises(TypeError):
            fit_rnmf_l1(self.V.astype(bool), 2, 0.1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V, 0, 0.1)
        with self.assertRaises(TypeError):
            fit_rnmf_l1(self.V, True, 0.1)
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V, 2, -0.1)
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V, 2, 0.1, max_iter=0)
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V, 2, 0.1, W_init=np.ones((10, 3)))
        with self.assertRaises(ValueError):
            fit_rnmf_l1(self.V, 2, 0.1, H_init=-np.ones((2, 8)))


if __name__ == "__main__":
    unittest.main()
