"""Member 3 composition checks, not an experiment runner.

Synthetic checks need no dataset files. The supplied-release checks use code/data/
located by tests/conftest.py and are skipped when the datasets are absent:
    pytest -q code/algorithm/tests/test_data_noise_integration.py

All block parameters below are TEST FIXTURE ONLY — NOT FINAL EXPERIMENT SETTINGS.
Figures render in memory using Agg and are explicitly closed without saving.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from utils.data_loader import load_data
from utils.occlusion import apply_occlusion
from utils.sampling import sample_columns
from utils.visualize import plot_occlusion_pair


SUBSET_SEED = 42
OCCLUSION_SEED = 314


class DataNoiseIntegrationTests(unittest.TestCase):
    def assert_pipeline(self, V_full, labels_full, image_shape, subset_count,
                        test_b, test_m, name):
        """Shared assertions for synthetic and opt-in supplied-data fixtures."""
        full_before, labels_before = V_full.copy(), labels_full.copy()
        self.assertEqual(labels_full.shape, (V_full.shape[1],))
        self.assertTrue(np.isfinite(V_full).all())
        self.assertTrue(np.all((V_full >= 0) & (V_full <= 1)))

        V_sub, labels_sub, indices = sample_columns(
            V_full, labels_full, fraction=0.9, seed=SUBSET_SEED
        )
        self.assertEqual(V_sub.shape, (V_full.shape[0], subset_count))
        self.assertEqual(labels_sub.shape, (subset_count,))
        self.assertEqual(indices.shape, (subset_count,))
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
        self.assertEqual(np.unique(indices).size, subset_count)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertTrue(np.all((indices >= 0) & (indices < V_full.shape[1])))
        np.testing.assert_array_equal(V_sub, V_full[:, indices])
        np.testing.assert_array_equal(labels_sub, labels_full[indices])
        np.testing.assert_array_equal(V_full, full_before)
        np.testing.assert_array_equal(labels_full, labels_before)
        self.assertFalse(np.shares_memory(V_sub, V_full))
        self.assertFalse(np.shares_memory(labels_sub, labels_full))

        sub_before, sub_labels_before = V_sub.copy(), labels_sub.copy()
        V_noisy, mask, positions = apply_occlusion(
            V_sub, image_shape, block_size=test_b, num_blocks=test_m,
            seed=OCCLUSION_SEED,
        )
        self.assertEqual(V_noisy.shape, V_sub.shape)
        self.assertEqual(mask.shape, V_sub.shape)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(positions.shape, (subset_count, test_m, 2))
        self.assertTrue(np.issubdtype(positions.dtype, np.integer))
        self.assertFalse(np.shares_memory(V_noisy, V_sub))
        self.assertFalse(np.shares_memory(V_noisy, V_full))
        self.assertTrue(np.isfinite(V_noisy).all())
        self.assertTrue(np.all((V_noisy >= 0) & (V_noisy <= 1)))
        np.testing.assert_array_equal(V_sub, sub_before)
        np.testing.assert_array_equal(V_full, full_before)
        np.testing.assert_array_equal(V_noisy[~mask], V_sub[~mask])
        self.assertTrue(np.all(V_noisy[mask] == 1.0))
        height, width = image_shape
        rows, cols = positions[:, :, 0], positions[:, :, 1]
        self.assertTrue(np.all((rows >= 0) & (rows <= height - test_b)))
        self.assertTrue(np.all((cols >= 0) & (cols <= width - test_b)))
        self.assertGreater(mask.sum(), 0)
        self.assertLessEqual(mask.sum(), subset_count * test_m * test_b**2)
        # Rebuild each union from the returned coordinates in C order.
        for j in range(subset_count):
            expected_mask = np.zeros(image_shape, dtype=bool)
            for row, col in positions[j]:
                expected_mask[row:row + test_b, col:col + test_b] = True
            np.testing.assert_array_equal(
                mask[:, j], expected_mask.reshape(-1, order="C")
            )

        noisy_before, mask_before = V_noisy.copy(), mask.copy()
        positions_before = positions.copy()
        # An empty temporary cwd detects accidental relative output. Mocking
        # savefig also checks the helper never attempts an implicit save.
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(Figure, "savefig") as save:
                    fig, axes = plot_occlusion_pair(
                        V_sub, V_noisy, image_shape, sample_index=0, mask=mask
                    )
                    try:
                        self.assertEqual(axes.shape, (3,))
                        self.assertEqual(len(fig.axes), 3)
                        for ax, matrix, title in zip(
                            axes, (V_sub, V_noisy, mask),
                            ("Clean", "Occluded", "Occlusion mask"),
                        ):
                            self.assertEqual(ax.get_title(), title)
                            self.assertEqual(len(ax.images), 1)
                            displayed = ax.images[0].get_array()
                            self.assertEqual(displayed.shape, image_shape)
                            np.testing.assert_array_equal(
                                displayed, matrix[:, 0].reshape(image_shape, order="C")
                            )
                        fig.canvas.draw()
                        save.assert_not_called()
                    finally:
                        plt.close(fig)
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(old_cwd)

        for actual, before in [
            (V_full, full_before), (labels_full, labels_before),
            (V_sub, sub_before), (labels_sub, sub_labels_before),
            (V_noisy, noisy_before), (mask, mask_before),
            (positions, positions_before),
        ]:
            np.testing.assert_array_equal(actual, before)

        repeated_sub, repeated_labels, repeated_indices = sample_columns(
            V_full, labels_full, fraction=0.9, seed=SUBSET_SEED
        )
        for actual, expected in zip(
            (repeated_sub, repeated_labels, repeated_indices),
            (V_sub, labels_sub, indices),
        ):
            np.testing.assert_array_equal(actual, expected)
        repeated_noise = apply_occlusion(
            repeated_sub, image_shape, test_b, test_m, seed=OCCLUSION_SEED
        )
        for actual, expected in zip(repeated_noise, (V_noisy, mask, positions)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(repeated_sub, sub_before)
        np.testing.assert_array_equal(V_full, full_before)
        np.testing.assert_array_equal(labels_full, labels_before)
        print(
            f"\n{name} integration: full={V_full.shape}, sampled={V_sub.shape}, "
            f"image_shape={image_shape}, test_b={test_b}, test_m={test_m}, "
            f"actual_coverage={mask.mean():.12g}, clean_unchanged=True, "
            "same_seeds_identical=True, three_panels_rendered=True"
        )

    def test_synthetic_pipeline(self):
        V = np.arange(480, dtype=np.float64).reshape(24, 20) / 480
        labels = np.repeat(np.arange(4), 5)
        self.assert_pipeline(V, labels, (4, 6), 18, 2, 2, "Synthetic")

    def test_subset_and_occlusion_seeds_control_separate_operations(self):
        V = np.arange(2400, dtype=np.float64).reshape(24, 100) / 2400
        labels = np.repeat(np.arange(10), 10)
        sub, _, indices = sample_columns(V, labels, seed=SUBSET_SEED)
        other_sub, _, other_indices = sample_columns(V, labels, seed=SUBSET_SEED + 1)
        self.assertFalse(np.array_equal(indices, other_indices))
        noisy, mask, positions = apply_occlusion(sub, (4, 6), 2, 2, OCCLUSION_SEED)
        # Same image dimensions/count and occlusion seed: changing the subset
        # changes the underlying images, but not the block placement arrays.
        _, other_mask, other_positions = apply_occlusion(
            other_sub, (4, 6), 2, 2, OCCLUSION_SEED
        )
        np.testing.assert_array_equal(other_positions, positions)
        np.testing.assert_array_equal(other_mask, mask)
        _, _, changed_positions = apply_occlusion(
            sub, (4, 6), 2, 2, OCCLUSION_SEED + 1
        )
        self.assertFalse(np.array_equal(changed_positions, positions))
        np.testing.assert_array_equal(sub, V[:, indices])
        self.assertFalse(np.shares_memory(noisy, sub))
        repeated_indices = sample_columns(V, labels, seed=SUBSET_SEED)[2]
        np.testing.assert_array_equal(repeated_indices, indices)

    @pytest.mark.usefixtures("supplied_data")
    def test_supplied_orl_pipeline(self):
        V, labels, shape = load_data(self.data_root / "ORL", reduce=3)
        self.assertEqual(V.shape, (1110, 400))
        self.assertEqual(shape, (37, 30))
        self.assert_pipeline(V, labels, shape, 360, 3, 2, "ORL")

    @pytest.mark.usefixtures("supplied_data")
    def test_supplied_yaleb_pipeline(self):
        V, labels, shape = load_data(self.data_root / "CroppedYaleB", reduce=4)
        self.assertEqual(V.shape, (2016, 2414))
        self.assertEqual(shape, (48, 42))
        self.assert_pipeline(V, labels, shape, 2172, 3, 2, "CroppedYaleB")


if __name__ == "__main__":
    unittest.main()
