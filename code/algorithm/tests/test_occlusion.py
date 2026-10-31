"""Member 3 occlusion tests; synthetic tests require only NumPy.

Run from the repository root (any directory works):
    pytest -q code/algorithm/tests/test_occlusion.py

The supplied-data checks use code/data/ located by tests/conftest.py and are
skipped when the datasets are absent.
"""

import unittest

import numpy as np
import pytest

from utils.occlusion import apply_occlusion


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (7, 11)
        self.V = np.arange(7 * 11 * 4, dtype=np.float64).reshape(77, 4) / 308

    def test_single_image_one_block_and_c_order(self):
        # Non-square image, unique pixels: row/column swaps cannot go unnoticed.
        V = np.arange(24, dtype=np.float64).reshape(24, 1) / 24
        before = V.copy()
        noisy, mask, positions = apply_occlusion(V, (4, 6), 2, 1, seed=42)
        self.assertEqual(noisy.shape, V.shape)
        self.assertEqual(mask.shape, V.shape)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(positions.shape, (1, 1, 2))
        self.assertTrue(np.issubdtype(positions.dtype, np.integer))
        row, col = positions[0, 0]
        self.assertTrue(0 <= row <= 2)
        self.assertTrue(0 <= col <= 4)
        # Independently calculate the flattened row-major indices.
        expected = sorted([row * 6 + col, row * 6 + col + 1,
                           (row + 1) * 6 + col, (row + 1) * 6 + col + 1])
        np.testing.assert_array_equal(np.flatnonzero(mask[:, 0]), expected)
        self.assertEqual(mask.sum(), 4)
        np.testing.assert_array_equal(noisy[mask], np.ones(4))
        np.testing.assert_array_equal(noisy[~mask], V[~mask])
        np.testing.assert_array_equal(V, before)

    def test_multiple_images_blocks_union_and_bounds(self):
        b, m = 3, 5
        noisy, mask, positions = apply_occlusion(self.V, self.shape, b, m, seed=7)
        height, width = self.shape
        self.assertEqual(positions.shape, (4, m, 2))
        self.assertTrue(np.all((positions[:, :, 0] >= 0) &
                               (positions[:, :, 0] <= height - b)))
        self.assertTrue(np.all((positions[:, :, 1] >= 0) &
                               (positions[:, :, 1] <= width - b)))
        for j in range(4):
            expected = np.zeros(height * width, dtype=bool)
            for row, col in positions[j]:
                for r in range(row, row + b):
                    expected[r * width + col:r * width + col + b] = True
            np.testing.assert_array_equal(mask[:, j], expected)
            self.assertLessEqual(mask[:, j].sum(), m * b * b)
        np.testing.assert_array_equal(noisy[mask], np.ones(mask.sum()))
        np.testing.assert_array_equal(noisy[~mask], self.V[~mask])

    def test_repeated_full_image_blocks_allow_overlap(self):
        # Only one valid position: repeated blocks must not trigger rejection.
        V = np.zeros((16, 3))
        noisy, mask, positions = apply_occlusion(V, (4, 4), 4, 3, seed=2)
        np.testing.assert_array_equal(positions, np.zeros((3, 3, 2), dtype=int))
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(mask.sum(axis=0), [16, 16, 16])
        np.testing.assert_array_equal(noisy, np.ones_like(V))

    def test_bottom_right_boundary_is_reachable(self):
        # All nine positions are legal in a 4x4 image with a 2x2 block.
        # A fixed large draw detects accidentally excluding the final row/col.
        _, mask, positions = apply_occlusion(np.zeros((16, 100)), (4, 4), 2, 1, 9)
        observed = {tuple(p) for p in positions[:, 0]}
        self.assertEqual(observed, {(r, c) for r in range(3) for c in range(3)})
        np.testing.assert_array_equal(mask.sum(axis=0), np.full(100, 4))

    def test_reproducibility_and_seed_variation(self):
        first = apply_occlusion(self.V, self.shape, 2, 4, seed=42)
        repeated = apply_occlusion(self.V, self.shape, 2, 4, seed=np.int64(42))
        for expected, actual in zip(first, repeated):
            np.testing.assert_array_equal(actual, expected)
        different = apply_occlusion(self.V, self.shape, 2, 4, seed=43)
        self.assertFalse(np.array_equal(first[2], different[2]))

    def test_global_rng_independence(self):
        saved = np.random.get_state()
        try:
            np.random.seed(1)
            expected = np.random.random(10)
            np.random.seed(1)
            first = apply_occlusion(self.V, self.shape, 2, 3, seed=42)
            np.testing.assert_array_equal(np.random.random(10), expected)
            np.random.seed(999)
            np.random.random(25)
            second = apply_occlusion(self.V, self.shape, 2, 3, seed=42)
            for expected_output, actual in zip(first, second):
                np.testing.assert_array_equal(actual, expected_output)
        finally:
            np.random.set_state(saved)

    def test_zero_blocks(self):
        noisy, mask, positions = apply_occlusion(self.V, self.shape, 2, 0, seed=42)
        np.testing.assert_array_equal(noisy, self.V)
        self.assertFalse(np.shares_memory(noisy, self.V))
        self.assertEqual(mask.shape, self.V.shape)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertFalse(mask.any())
        self.assertEqual(positions.shape, (4, 0, 2))
        self.assertTrue(np.issubdtype(positions.dtype, np.integer))
        noisy[:] = 0
        self.assertTrue(np.any(self.V != 0))

    def test_outputs_have_independent_storage(self):
        V = self.V[:, ::2]  # Non-contiguous, read-only input also works.
        before = V.copy()
        V.setflags(write=False)
        noisy, mask, positions = apply_occlusion(V, self.shape, 2, 3, seed=7)
        again = apply_occlusion(V, self.shape, 2, 3, seed=7)
        self.assertFalse(np.shares_memory(noisy, V))
        for output, other in zip((noisy, mask, positions), again):
            self.assertFalse(np.shares_memory(output, other))
        noisy[:] = 0
        mask[:] = False
        positions[:] = -1
        np.testing.assert_array_equal(V, before)
        self.assertTrue(again[1].any())
        self.assertTrue(np.all(again[2] >= 0))

    def test_mask_includes_already_white_pixels(self):
        V = np.ones((16, 1))
        noisy, mask, _ = apply_occlusion(V, (4, 4), 2, 1, seed=3)
        np.testing.assert_array_equal(noisy, V)
        self.assertEqual(mask.sum(), 4)  # Coverage, not a numerical-change mask.

    def test_dtypes_and_numpy_integer_parameters(self):
        for dtype in [np.float32, np.float64, np.int16, np.uint8]:
            with self.subTest(dtype=dtype):
                V = np.zeros((6, 2), dtype=dtype)
                noisy, _, _ = apply_occlusion(
                    V, (np.int64(2), np.int64(3)), np.int64(1), np.int64(1),
                    np.int64(0),
                )
                expected = dtype if np.issubdtype(dtype, np.floating) else np.float64
                self.assertEqual(noisy.dtype, np.dtype(expected))

    def test_zero_columns(self):
        noisy, mask, positions = apply_occlusion(np.empty((6, 0)), (2, 3), 1, 2, 0)
        self.assertEqual(noisy.shape, (6, 0))
        self.assertEqual(mask.shape, (6, 0))
        self.assertEqual(positions.shape, (0, 2, 2))

    def test_invalid_matrix_dimensions_and_types(self):
        for V in [np.zeros(6), np.zeros((1, 2, 3))]:
            with self.subTest(shape=V.shape), self.assertRaisesRegex(ValueError, "2D"):
                apply_occlusion(V, (2, 3), 1, 1, 0)
        for dtype in [bool, complex, object, str]:
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "numeric"):
                apply_occlusion(np.zeros((6, 1), dtype=dtype), (2, 3), 1, 1, 0)

    def test_invalid_matrix_values(self):
        for value in [np.nan, np.inf, -np.inf, -0.01, 1.01]:
            V = np.zeros((6, 1))
            V[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_occlusion(V, (2, 3), 1, 1, 0)

    def test_invalid_image_shape(self):
        for shape in [(2,), (2, 3, 1), (2, 4), (0, 3), (-2, 3), (2, 0), (2, -3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                apply_occlusion(np.zeros((6, 1)), shape, 1, 1, 0)
        for shape in [None, 6, (2.0, 3), (2, 3.0), (True, 6), (2, np.bool_(True))]:
            with self.subTest(shape=shape), self.assertRaises(TypeError):
                apply_occlusion(np.zeros((6, 1)), shape, 1, 1, 0)

    def test_invalid_block_size(self):
        for b in [0, -1, 8, 12]:
            with self.subTest(b=b), self.assertRaises(ValueError):
                apply_occlusion(self.V, self.shape, b, 1, 0)
        for b in [True, np.bool_(True), 2.0, "2", None]:
            with self.subTest(b=b), self.assertRaises(TypeError):
                apply_occlusion(self.V, self.shape, b, 1, 0)
        with self.assertRaisesRegex(ValueError, "both height and width"):
            apply_occlusion(np.zeros((12, 1)), (6, 2), 3, 1, 0)

    def test_invalid_num_blocks(self):
        with self.assertRaisesRegex(ValueError, "num_blocks"):
            apply_occlusion(self.V, self.shape, 2, -1, 0)
        for m in [True, np.bool_(False), 1.0, "1", None]:
            with self.subTest(m=m), self.assertRaisesRegex(TypeError, "num_blocks"):
                apply_occlusion(self.V, self.shape, 2, m, 0)

    def test_invalid_seed_even_with_zero_blocks(self):
        for m in [0, 1]:
            with self.subTest(m=m):
                with self.assertRaisesRegex(ValueError, "seed"):
                    apply_occlusion(self.V, self.shape, 2, m, -1)
                for seed in [None, True, np.bool_(False), 1.0, "1", [1, 2],
                             np.random.default_rng(0)]:
                    with self.subTest(seed=seed), self.assertRaisesRegex(TypeError, "seed"):
                        apply_occlusion(self.V, self.shape, 2, m, seed)


@pytest.mark.usefixtures("supplied_data")
class SuppliedOcclusionTests(unittest.TestCase):
    def test_supplied_data_sanity(self):
        from utils.data_loader import load_data

        root = self.data_root
        for dataset, reduce, shape in [("ORL", 3, (37, 30)),
                                       ("CroppedYaleB", 4, (48, 42))]:
            with self.subTest(dataset=dataset):
                V, _, image_shape = load_data(root / dataset, reduce=reduce)
                self.assertEqual(image_shape, shape)
                before = V.copy()
                # Test fixture only: these are NOT final experimental settings.
                noisy, mask, positions = apply_occlusion(V, shape, 3, 2, seed=42)
                self.assertEqual(noisy.shape, V.shape)
                self.assertEqual(mask.shape, V.shape)
                self.assertEqual(mask.dtype, np.dtype(bool))
                self.assertEqual(positions.shape, (V.shape[1], 2, 2))
                self.assertTrue(np.isfinite(noisy).all())
                self.assertTrue(np.all((noisy >= 0) & (noisy <= 1)))
                np.testing.assert_array_equal(V, before)
                np.testing.assert_array_equal(noisy[~mask], V[~mask])
                self.assertTrue(np.all(noisy[mask] == 1.0))
                self.assertFalse(np.shares_memory(noisy, V))
                print(f"\n{dataset}: noisy={noisy.shape}, mask={mask.shape}, "
                      f"positions={positions.shape}, clean_unchanged=True")


if __name__ == "__main__":
    unittest.main()
