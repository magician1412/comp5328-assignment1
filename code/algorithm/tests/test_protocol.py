"""Checks that main.py encodes the agreed experimental protocol.

The protocol is fixed for the final experiment; any change to these values
must be deliberate, so they are pinned here. No dataset files are needed.
"""

import unittest

import numpy as np

import main
from nmf.nmf_baseline import nmf_baseline


class ProtocolTests(unittest.TestCase):
    def test_shared_protocol(self):
        self.assertEqual(main.SETTINGS,
                         [(4, 0), (4, 1), (8, 1), (12, 1), (16, 1), (8, 2), (8, 3)])
        self.assertEqual(main.REPEATS, 5)
        self.assertEqual(main.SAMPLE_FRACTION, 0.9)
        self.assertEqual((main.SAMPLE_SEED, main.CORRUPTION_SEED, main.INIT_SEED),
                         (100, 200, 300))

    def test_per_dataset_solver_settings(self):
        expected = {"orl": (1e-3, 0.01), "yaleb": (5e-4, 0.03)}
        for name, (nmf_tol, lam) in expected.items():
            with self.subTest(dataset=name):
                config = main.DATASET_CONFIGS[name]
                self.assertEqual(config["nmf"],
                                 {"tol": nmf_tol, "max_iter": 2000, "min_iter": 200})
                self.assertEqual(config["rnmf"],
                                 {"lam": lam, "tol": 0.0, "max_iter": 2000})

    def test_nmf_min_iter_matches_solver(self):
        # With a huge tolerance the stopping rule fires at the first allowed
        # iteration, which is the solver's built-in minimum.
        rng = np.random.default_rng(0)
        _, _, history, converged = nmf_baseline(rng.random((12, 10)), rank=2,
                                                max_iter=1000, tol=1e9)
        self.assertTrue(converged)
        self.assertEqual(len(history), main.NMF_MIN_ITER)


if __name__ == "__main__":
    unittest.main()
