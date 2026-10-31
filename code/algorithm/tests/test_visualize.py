"""Member 3 input-image visualization tests using the non-interactive Agg backend.

Run from the repository root (any directory works):
    pytest -q code/algorithm/tests/test_visualize.py

The supplied-data checks use code/data/ located by tests/conftest.py and are
skipped when the datasets are absent.

Saved figures are confined to automatically cleaned temporary directories.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest

from utils.occlusion import apply_occlusion
from utils.visualize import column_to_image, plot_occlusion_pair


class ColumnToImageTests(unittest.TestCase):
    def test_non_square_c_order_and_copy(self):
        V = np.column_stack([np.arange(6), np.arange(10, 16)])
        before = V.copy()
        image = column_to_image(V, (np.int64(2), np.int64(3)), np.int64(1))
        np.testing.assert_array_equal(image, [[10, 11, 12], [13, 14, 15]])
        self.assertEqual(image.dtype, V.dtype)
        self.assertFalse(np.shares_memory(image, V))
        image[:] = -1
        np.testing.assert_array_equal(V, before)

    def test_boolean_mask_reconstruction(self):
        V = np.array([[True], [False], [False], [True], [True], [False]])
        image = column_to_image(V, (2, 3), 0)
        np.testing.assert_array_equal(image, [[True, False, False], [True, True, False]])
        self.assertEqual(image.dtype, np.dtype(bool))

    def test_invalid_matrix_or_shape(self):
        for V in [np.zeros(6), np.zeros((1, 2, 3))]:
            with self.subTest(V_shape=V.shape), self.assertRaisesRegex(ValueError, "2D"):
                column_to_image(V, (2, 3), 0)
        for shape in [(2,), (2, 3, 1), (2, 4), (0, 3), (2, -3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                column_to_image(np.zeros((6, 1)), shape, 0)
        for shape in [None, 6, (2.0, 3), (2, 3.0), (True, 6), (2, np.bool_(True))]:
            with self.subTest(shape=shape), self.assertRaises(TypeError):
                column_to_image(np.zeros((6, 1)), shape, 0)

    def test_invalid_sample_indices_and_empty_dataset(self):
        V = np.zeros((6, 2))
        for index in [-1, 2]:
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "sample_index"):
                column_to_image(V, (2, 3), index)
        for index in [True, np.bool_(False), 1.0, "1", None]:
            with self.subTest(index=index), self.assertRaisesRegex(TypeError, "sample_index"):
                column_to_image(V, (2, 3), index)
        with self.assertRaisesRegex(ValueError, "sample_index"):
            column_to_image(np.empty((6, 0)), (2, 3), 0)


class PlotOcclusionPairTests(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 6)
        self.clean = np.arange(48, dtype=np.float64).reshape(24, 2) / 48
        self.noisy, self.mask, self.positions = apply_occlusion(
            self.clean, self.shape, 2, 2, seed=42
        )

    def tearDown(self):
        plt.close("all")

    def test_pair_panels_grayscale_and_title(self):
        fig, axes = plot_occlusion_pair(
            self.clean, self.noisy, self.shape, 1, title="Synthetic sanity check"
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual(axes.shape, (2,))
        self.assertEqual(len(fig.axes), 2)
        for ax, V, title in zip(axes, [self.clean, self.noisy], ["Clean", "Occluded"]):
            self.assertIsInstance(ax, Axes)
            self.assertEqual(ax.get_title(), title)
            self.assertEqual(len(ax.images), 1)
            artist = ax.images[0]
            self.assertEqual(artist.get_cmap().name, "gray")
            self.assertEqual(artist.get_clim(), (0, 1))
            self.assertEqual(artist.origin, "upper")
            self.assertEqual(artist.get_interpolation(), "nearest")
            self.assertEqual(len(ax.get_xticks()), 0)
            self.assertEqual(len(ax.get_yticks()), 0)
            np.testing.assert_array_equal(artist.get_array(), V[:, 1].reshape(4, 6, order="C"))
        self.assertEqual(fig._suptitle.get_text(), "Synthetic sanity check")

    def test_mask_panel_matches_occlusion_coordinates(self):
        _, axes = plot_occlusion_pair(self.clean, self.noisy, self.shape, 1, mask=self.mask)
        self.assertEqual(axes.shape, (3,))
        self.assertEqual(axes[2].get_title(), "Occlusion mask")
        displayed = axes[2].images[0].get_array()
        expected = np.zeros(self.shape, dtype=bool)
        for row, col in self.positions[1]:
            expected[row:row + 2, col:col + 2] = True
        np.testing.assert_array_equal(displayed, expected)
        np.testing.assert_array_equal(displayed, self.mask[:, 1].reshape(self.shape, order="C"))
        self.assertEqual(axes[2].images[0].get_clim(), (0, 1))

    def test_saves_png_to_exact_path_with_new_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = root / "keep.txt"
            unrelated.write_text("unchanged")
            path = root / "nested" / "pair.png"
            plot_occlusion_pair(self.clean, self.noisy, self.shape, 0, save_path=path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(unrelated.read_text(), "unchanged")
            self.assertEqual({p.relative_to(root) for p in root.rglob("*") if p.is_file()},
                             {Path("keep.txt"), Path("nested/pair.png")})

    def test_extensionless_save_uses_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair"
            plot_occlusion_pair(self.clean, self.noisy, self.shape, 0, save_path=path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_no_implicit_save_or_show(self):
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(Figure, "savefig") as save, patch.object(plt, "show") as show:
                    plot_occlusion_pair(self.clean, self.noisy, self.shape, 0)
                    save.assert_not_called()
                    show.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(old_cwd)

    def test_input_preservation(self):
        arrays = [self.clean, self.noisy, self.mask]
        before = [V.copy() for V in arrays]
        for V in arrays:
            V.setflags(write=False)
        fig, axes = plot_occlusion_pair(*arrays[:2], self.shape, 0, mask=self.mask)
        fig.canvas.draw()
        for ax in axes:
            ax.images[0].set_data(np.zeros(self.shape))
        for V, original in zip(arrays, before):
            np.testing.assert_array_equal(V, original)

    def test_invalid_matrix_shapes(self):
        for clean, noisy in [(self.clean[:, 0], self.noisy),
                             (self.clean, self.noisy[:, 0]),
                             (self.clean, self.noisy[:, :1])]:
            with self.subTest(clean=clean.shape, noisy=noisy.shape):
                with self.assertRaises(ValueError):
                    plot_occlusion_pair(clean, noisy, self.shape, 0)
        self.assertEqual(plt.get_fignums(), [])

    def test_plot_validates_shape_and_sample_index(self):
        for shape, index in [((3, 7), 0), ((0, 6), 0), (self.shape, -1), (self.shape, 2)]:
            with self.subTest(shape=shape, index=index), self.assertRaises(ValueError):
                plot_occlusion_pair(self.clean, self.noisy, shape, index)
        with self.assertRaises(TypeError):
            plot_occlusion_pair(self.clean, self.noisy, self.shape, True)
        self.assertEqual(plt.get_fignums(), [])

    def test_invalid_values_in_either_matrix(self):
        for which in [0, 1]:
            for value in [np.nan, np.inf, -np.inf, -0.1, 1.1]:
                arrays = [self.clean.copy(), self.noisy.copy()]
                arrays[which][0, 1] = value  # Validate even non-displayed columns.
                with self.subTest(which=which, value=value), self.assertRaises(ValueError):
                    plot_occlusion_pair(*arrays, self.shape, 0)
            for dtype in [bool, complex, object, str]:
                arrays = [self.clean.copy(), self.noisy.copy()]
                arrays[which] = arrays[which].astype(dtype)
                with self.subTest(which=which, dtype=dtype), self.assertRaises(TypeError):
                    plot_occlusion_pair(*arrays, self.shape, 0)
        self.assertEqual(plt.get_fignums(), [])

    def test_invalid_mask(self):
        with self.assertRaisesRegex(ValueError, "mask.*shape"):
            plot_occlusion_pair(self.clean, self.noisy, self.shape, 0, mask=self.mask[:, :1])
        for dtype in [int, float, str, object]:
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "boolean"):
                plot_occlusion_pair(self.clean, self.noisy, self.shape, 0,
                                    mask=self.mask.astype(dtype))
        self.assertEqual(plt.get_fignums(), [])

    def test_failed_save_closes_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.png"
            with patch.object(Figure, "savefig", side_effect=OSError("save failed")):
                with self.assertRaisesRegex(OSError, "save failed"):
                    plot_occlusion_pair(self.clean, self.noisy, self.shape, 0, save_path=path)
        self.assertEqual(plt.get_fignums(), [])


@pytest.mark.usefixtures("supplied_data")
class SuppliedVisualizationTests(unittest.TestCase):
    def test_supplied_image_shapes(self):
        from utils.data_loader import load_data

        root = self.data_root
        for dataset, reduce, expected in [("ORL", 3, (37, 30)),
                                          ("CroppedYaleB", 4, (48, 42))]:
            with self.subTest(dataset=dataset):
                V, _, shape = load_data(root / dataset, reduce=reduce)
                clean = V[:, :1]
                # Sanity fixture only, not a final experimental setting.
                noisy, mask, _ = apply_occlusion(clean, shape, 3, 1, seed=42)
                fig, axes = plot_occlusion_pair(clean, noisy, shape, 0, mask=mask)
                try:
                    fig.canvas.draw()
                    for ax in axes:
                        self.assertEqual(ax.images[0].get_array().shape, expected)
                    print(f"\n{dataset}: clean/noisy/mask displayed shape={expected}")
                finally:
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main()
