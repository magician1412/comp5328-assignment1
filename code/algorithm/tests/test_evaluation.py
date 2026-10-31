"""Tests for the masked RRE, paired differences, ddof=1 statistics and traces."""

import math
import unittest

import numpy as np

from experiments.experiment import (
    aggregate_rows,
    objective_trace_rows,
    paired_differences,
    sample_std,
)
from utils.evaluation_metrics import (
    clustering_scores,
    masked_relative_reconstruction_error,
    relative_reconstruction_error,
)


class MaskedRRETests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.V = rng.random((6, 5)) + 0.1
        self.W = rng.random((6, 2))
        self.H = rng.random((2, 5))
        self.mask = np.zeros((6, 5), dtype=bool)
        self.mask[:2, :3] = True

    def test_matches_definition(self):
        E = self.V - self.W @ self.H
        expected = np.linalg.norm(E[self.mask]) / np.linalg.norm(self.V[self.mask])
        self.assertAlmostEqual(
            masked_relative_reconstruction_error(self.V, self.W, self.H, self.mask),
            expected)

    def test_full_mask_equals_overall_rre(self):
        full = np.ones_like(self.mask)
        self.assertAlmostEqual(
            masked_relative_reconstruction_error(self.V, self.W, self.H, full),
            relative_reconstruction_error(self.V, self.W, self.H))

    def test_empty_mask_gives_nan(self):
        empty = np.zeros_like(self.mask)
        self.assertTrue(math.isnan(
            masked_relative_reconstruction_error(self.V, self.W, self.H, empty)))

    def test_rejects_non_boolean_or_misshaped_mask(self):
        with self.assertRaises(ValueError):
            masked_relative_reconstruction_error(self.V, self.W, self.H,
                                                 self.mask.astype(int))
        with self.assertRaises(ValueError):
            masked_relative_reconstruction_error(self.V, self.W, self.H,
                                                 self.mask[:, :2])


def _row(method, block_size, num_blocks, repeat, rre):
    """Return a minimal raw-results row."""
    return {
        "dataset": "toy", "reduce": 1, "algorithm": method, "rank": 2,
        "lambda": None, "max_iter": 10, "tol": 0.0, "block_size": block_size,
        "num_blocks": num_blocks, "sample_fraction": 0.9, "n_features": 4,
        "n_samples": 3, "repeat": repeat, "corrupted_fraction": 0.1, "rre": rre,
        "rre_occluded": float("nan"), "rre_unoccluded": rre, "accuracy": 1.0,
        "nmi": 1.0, "runtime_seconds": 0.1, "iterations": 10, "converged": True,
        "objective_final": 1.0,
    }


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row("Frobenius-NMF", 8, 2, 0, 0.40), _row("RNMF-L1", 8, 2, 0, 0.25),
            _row("Frobenius-NMF", 8, 2, 1, 0.50), _row("RNMF-L1", 8, 2, 1, 0.30),
            _row("Frobenius-NMF", 8, 2, 2, 0.45), _row("RNMF-L1", 8, 2, 2, 0.35),
        ]

    def test_sample_std_uses_ddof_1(self):
        self.assertAlmostEqual(sample_std([1.0, 2.0, 3.0]), 1.0)
        self.assertTrue(math.isnan(sample_std([1.0])))

    def test_paired_differences_by_repeat(self):
        diff = paired_differences(self.rows)[("toy", 8, 2)]
        np.testing.assert_allclose(diff, [0.15, 0.20, 0.10])

    def test_summary_carries_paired_difference_and_ddof_1_std(self):
        summary = aggregate_rows(self.rows)
        self.assertEqual(len(summary), 2)
        for row in summary:
            self.assertAlmostEqual(row["paired_rre_diff_mean"], 0.15)
            self.assertAlmostEqual(row["paired_rre_diff_std"], 0.05)
            self.assertEqual(row["paired_rre_diff_n"], 3)
            self.assertTrue(math.isnan(row["rre_occluded_mean"]))
        nmf = next(r for r in summary if r["algorithm"] == "Frobenius-NMF")
        self.assertAlmostEqual(nmf["rre_std"], np.std([0.40, 0.50, 0.45], ddof=1))


class ObjectiveTraceTests(unittest.TestCase):
    def test_long_format(self):
        records = {(8, 2): {"objectives": {"Frobenius-NMF": [3.0, 2.0],
                                           "RNMF-L1": [5.0]}}}
        rows = objective_trace_rows(records)
        self.assertEqual(rows[0], {"setting": "8:2", "block_size": 8, "num_blocks": 2,
                                   "method": "Frobenius-NMF", "iteration": 1,
                                   "objective": 3.0})
        self.assertEqual([(r["method"], r["iteration"]) for r in rows],
                         [("Frobenius-NMF", 1), ("Frobenius-NMF", 2), ("RNMF-L1", 1)])


class ClusteringTests(unittest.TestCase):
    def test_kmeans_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(1)
        H = np.hstack([rng.random((3, 6)), rng.random((3, 6)) + 2.0])
        labels = np.repeat([0, 1], 6)
        first = clustering_scores(H, labels, seed=300, n_init=10)
        second = clustering_scores(H, labels, seed=300, n_init=10)
        np.testing.assert_array_equal(first["cluster_ids"], second["cluster_ids"])
        self.assertEqual(first["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
