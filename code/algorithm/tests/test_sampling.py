"""Member 3 sampling tests; normal unit tests require only NumPy.

From the repository root:
    pytest -q code/algorithm/tests/test_sampling.py

The supplied-dataset checks (which also require Pillow) use code/data/ located by
tests/conftest.py and are skipped when the datasets are absent.
"""

import unittest

import numpy as np
import pytest

from utils.sampling import sample_columns


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.V = np.arange(500, dtype=np.float64).reshape(5, 100)
        self.labels = np.repeat(np.arange(10), 10)

    def test_shape_indices_and_alignment(self):
        V_sub, labels_sub, indices = sample_columns(self.V, self.labels)
        self.assertEqual(V_sub.shape, (5, 90))
        self.assertEqual(labels_sub.shape, (90,))
        self.assertEqual(indices.shape, (90,))
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
        self.assertEqual(np.unique(indices).size, 90)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertTrue(np.all((indices >= 0) & (indices < 100)))
        np.testing.assert_array_equal(V_sub, self.V[:, indices])
        np.testing.assert_array_equal(labels_sub, self.labels[indices])

    def test_floor_count_and_one_column(self):
        V_sub, labels_sub, indices = sample_columns(
            self.V[:, :11], self.labels[:11], fraction=0.9
        )
        self.assertEqual(V_sub.shape, (5, 9))
        self.assertEqual(labels_sub.size, 9)
        self.assertEqual(indices.size, 9)
        V_sub, labels_sub, indices = sample_columns([[7]], [3], fraction=1)
        np.testing.assert_array_equal(V_sub, [[7]])
        np.testing.assert_array_equal(labels_sub, [3])
        np.testing.assert_array_equal(indices, [0])

    def test_reproducibility_and_seed_variation(self):
        first = sample_columns(self.V, self.labels, seed=42)
        repeated = sample_columns(self.V, self.labels, seed=np.int64(42))
        for expected, actual in zip(first, repeated):
            np.testing.assert_array_equal(actual, expected)
        different = sample_columns(self.V, self.labels, seed=43)
        self.assertFalse(np.array_equal(first[2], different[2]))

    def test_global_rng_independence(self):
        # Save/restore legacy global RNG state so this test has no lasting effect.
        saved = np.random.get_state()
        try:
            np.random.seed(1)
            expected_draws = np.random.random(10)
            np.random.seed(1)
            first = sample_columns(self.V, self.labels, seed=42)
            np.testing.assert_array_equal(np.random.random(10), expected_draws)
            np.random.seed(999)
            np.random.random(25)
            second = sample_columns(self.V, self.labels, seed=42)
            for expected, actual in zip(first, second):
                np.testing.assert_array_equal(actual, expected)
        finally:
            np.random.set_state(saved)

    def test_global_sampling_is_label_agnostic(self):
        # Membership must not depend on class sizes, names, or even class count.
        _, _, expected = sample_columns(self.V, self.labels, seed=8)
        for labels in [np.zeros(100), np.arange(100),
                       np.array(["majority"] * 99 + ["minority"])]:
            with self.subTest(label_dtype=labels.dtype):
                V_sub, labels_sub, indices = sample_columns(self.V, labels, seed=8)
                np.testing.assert_array_equal(indices, expected)
                np.testing.assert_array_equal(V_sub, self.V[:, expected])
                np.testing.assert_array_equal(labels_sub, labels[expected])

    def test_full_fraction_preserves_order(self):
        V_sub, labels_sub, indices = sample_columns(
            self.V, self.labels, fraction=1.0, seed=9
        )
        np.testing.assert_array_equal(indices, np.arange(100))
        np.testing.assert_array_equal(V_sub, self.V)
        np.testing.assert_array_equal(labels_sub, self.labels)

    def test_output_mutation_preserves_clean_inputs(self):
        for fraction in [0.9, 1.0]:
            with self.subTest(fraction=fraction):
                # A non-contiguous input must be safe too.
                V = self.V[:, ::2]
                labels = self.labels[::2]
                before_V, before_labels = V.copy(), labels.copy()
                V_sub, labels_sub, _ = sample_columns(V, labels, fraction=fraction)
                self.assertFalse(np.shares_memory(V_sub, V))
                self.assertFalse(np.shares_memory(labels_sub, labels))
                V_sub[:] = -1
                labels_sub[:] = -1
                np.testing.assert_array_equal(V, before_V)
                np.testing.assert_array_equal(labels, before_labels)

    def test_preserves_dtypes(self):
        V_sub, labels_sub, _ = sample_columns(
            self.V.astype(np.float32), self.labels.astype(np.int16)
        )
        self.assertEqual(V_sub.dtype, np.dtype(np.float32))
        self.assertEqual(labels_sub.dtype, np.dtype(np.int16))

    def test_invalid_fractions(self):
        for fraction in [0, -0.1, 1.1, np.nan, np.inf, -np.inf]:
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "fraction must be finite"):
                    sample_columns(self.V, self.labels, fraction=fraction)
        for fraction in ["0.9", None, True, np.bool_(False), 0.9 + 0j, [0.9]]:
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(TypeError, "real numeric scalar"):
                    sample_columns(self.V, self.labels, fraction=fraction)

    def test_zero_result_and_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "at least one column"):
            sample_columns(self.V, self.labels, fraction=0.001)
        with self.assertRaisesRegex(ValueError, "at least one column"):
            sample_columns(np.empty((5, 0)), np.empty(0), fraction=1)

    def test_invalid_shapes(self):
        for V, labels, message in [
            (np.zeros(100), self.labels, "2D matrix"),
            (np.zeros((1, 5, 100)), self.labels, "2D matrix"),
            (self.V, self.labels.reshape(1, -1), "1D array"),
            (self.V, np.array(1), "1D array"),
            (self.V, self.labels[:-1], "labels length"),
        ]:
            with self.subTest(V_shape=V.shape, labels_shape=labels.shape):
                with self.assertRaisesRegex(ValueError, message):
                    sample_columns(V, labels)

    def test_invalid_seeds(self):
        for seed in [-1, np.int64(-5)]:
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                sample_columns(self.V, self.labels, seed=seed)
        for seed in [None, 1.5, "42", True, np.bool_(True),
                     np.random.default_rng(42), [1, 2]]:
            with self.subTest(seed=seed), self.assertRaisesRegex(TypeError, "seed"):
                # Validation also applies when sampling every column.
                sample_columns(self.V, self.labels, fraction=1, seed=seed)


@pytest.mark.usefixtures("supplied_data")
class SuppliedSamplingTests(unittest.TestCase):
    def test_supplied_subset_sizes(self):
        # Keep Pillow/image loading out of the normal sampling-only test path.
        from utils.data_loader import load_data

        root = self.data_root
        for dataset, reduce, count, subset in [
            ("ORL", 3, 400, 360), ("CroppedYaleB", 4, 2414, 2172),
        ]:
            with self.subTest(dataset=dataset):
                V, labels, _ = load_data(root / dataset, reduce=reduce)
                self.assertEqual(V.shape[1], count)
                V_sub, labels_sub, indices = sample_columns(V, labels, seed=42)
                self.assertEqual(V_sub.shape, (V.shape[0], subset))
                self.assertEqual(labels_sub.shape, (subset,))
                self.assertEqual(indices.shape, (subset,))
                self.assertTrue(np.all(np.diff(indices) > 0))
                self.assertTrue(np.all((indices >= 0) & (indices < count)))
                np.testing.assert_array_equal(V_sub, V[:, indices])
                np.testing.assert_array_equal(labels_sub, labels[indices])
                repeated = sample_columns(V, labels, seed=42)
                for expected, actual in zip((V_sub, labels_sub, indices), repeated):
                    np.testing.assert_array_equal(actual, expected)
                different = sample_columns(V, labels, seed=43)[2]
                self.assertFalse(np.array_equal(indices, different))
                print(
                    f"\n{dataset}: {count} -> {subset} columns, "
                    f"V_sub={V_sub.shape}, labels_sub={labels_sub.shape}, "
                    "same_seed_identical=True, different_seed_membership=True"
                )


if __name__ == "__main__":
    unittest.main()
