"""Tests for the noise montage figure and its faithfulness to main.py.

Run from the repository root (any directory works):
    pytest -q code/algorithm/tests/test_noise_montage.py

All checks use tiny synthetic matrices or parse main.py; no dataset files are
needed and no figure is written to disk.
"""

import ast
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_noise_montage as montage
from experiments.experiment import run_condition
from utils.visualize import plot_occlusion_montage

MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"
# Large enough for the biggest experiment block (b = 16) in both dimensions.
SHAPE = (18, 17)


def synthetic_data(n=10, seed=0):
    """Return (V, labels) with values in [0, 0.9] so occlusion (1.0) is visible."""
    rng = np.random.default_rng(seed)
    V = rng.random((SHAPE[0] * SHAPE[1], n)) * 0.9
    labels = np.arange(n) % 2
    return V, labels


def main_module_literals():
    """Return {name: value} for the module-level literal constants of main.py."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass  # Not a pure literal (for example it refers to NMF_MIN_ITER).
    return values


def main_dataset_configs():
    """Return {dataset: {"folder", "reduce"}} parsed from main.DATASET_CONFIGS."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "DATASET_CONFIGS")
    configs = {}
    for key, value in zip(node.value.keys, node.value.values):
        fields = {ast.literal_eval(k): v for k, v in zip(value.keys, value.values)}
        configs[ast.literal_eval(key)] = {
            "folder": ast.literal_eval(fields["folder"]),
            "reduce": ast.literal_eval(fields["reduce"]),
        }
    return configs


class MainPyConstantsTests(unittest.TestCase):
    def test_duplicated_constants_match_main_py(self):
        values = main_module_literals()
        self.assertEqual([tuple(s) for s in values["SETTINGS"]], montage.SETTINGS)
        self.assertEqual(values["SAMPLE_FRACTION"], montage.SAMPLE_FRACTION)
        self.assertEqual(values["SAMPLE_SEED"], montage.SAMPLE_SEED)
        self.assertEqual(values["CORRUPTION_SEED"], montage.CORRUPTION_SEED)
        self.assertEqual(values["VISUAL_SAMPLE_INDEX"], montage.SAMPLE_INDEX)
        configs = main_dataset_configs()
        self.assertEqual(set(configs), set(montage.DATASETS))
        for name, config in montage.DATASETS.items():
            with self.subTest(dataset=name):
                self.assertEqual(configs[name]["folder"], config["folder"])
                self.assertEqual(configs[name]["reduce"], config["reduce"])


def main_repeats_default():
    """Return main.py's REPEATS constant, the default of its --repeats option."""
    return main_module_literals()["REPEATS"]


class CoverageTests(unittest.TestCase):
    def test_coverage_repeats_match_main_py_default(self):
        self.assertEqual(montage.COVERAGE_REPEATS, main_repeats_default())

    def test_coverage_matches_block_area(self):
        V, labels = synthetic_data()
        pixels = SHAPE[0] * SHAPE[1]
        coverage = montage.coverage_by_setting(V, labels, SHAPE)
        self.assertEqual(list(coverage), montage.SETTINGS)
        for (block_size, num_blocks), values in coverage.items():
            with self.subTest(b=block_size, m=num_blocks):
                self.assertEqual(len(values), montage.COVERAGE_REPEATS)
                nominal = block_size ** 2 / pixels
                if num_blocks == 0:
                    self.assertTrue(np.all(values == 0.0))
                elif num_blocks == 1:
                    # A single block always lies fully inside the image.
                    self.assertTrue(np.all(values == nominal), values)
                else:
                    # Overlapping blocks can only reduce the covered area.
                    self.assertTrue(np.all(values <= num_blocks * nominal), values)

    def test_coverage_rows_report_mean_and_sample_std_in_percent(self):
        rows = montage.coverage_rows("synthetic", {(8, 2): np.array([0.1, 0.2, 0.3])})
        self.assertEqual(rows, [{
            "dataset": "synthetic", "block_size": 8, "num_blocks": 2, "repeats": 3,
            "mean_coverage_pct": "20.00", "std_coverage_pct": "10.00",
        }])


class FaithfulnessTests(unittest.TestCase):
    def test_noise_matches_experiment_run_condition(self):
        V, labels = synthetic_data()
        for repeat in (0, 1):
            for block_size, num_blocks in montage.SETTINGS:
                with self.subTest(repeat=repeat, b=block_size, m=num_blocks):
                    rows, artifacts = run_condition(
                        V, labels, SHAPE, [], dataset="synthetic", reduce=1, rank=2,
                        block_size=block_size, num_blocks=num_blocks, repeat=repeat,
                        sample_fraction=montage.SAMPLE_FRACTION,
                        sample_seed=montage.SAMPLE_SEED,
                        corruption_seed=montage.CORRUPTION_SEED,
                        clustering=False, return_artifacts=True,
                    )
                    self.assertEqual(rows, [])
                    V_sub, V_noisy, mask = montage.experiment_noise(
                        V, labels, SHAPE, block_size, num_blocks, repeat)
                    np.testing.assert_array_equal(V_sub, artifacts["clean"])
                    np.testing.assert_array_equal(V_noisy, artifacts["occluded"])
                    self.assertEqual(bool(mask.any()), num_blocks > 0)


class MontageFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_montage_has_seven_image_axes_of_the_same_face(self):
        V, labels = synthetic_data()
        fig, axes = montage.make_montage(V, labels, SHAPE)
        self.assertEqual(len(axes), 7)
        self.assertEqual(len(fig.axes), 7)
        V_sub, _, _ = montage.experiment_noise(V, labels, SHAPE, 4, 0)
        clean = V_sub[:, montage.SAMPLE_INDEX].reshape(SHAPE)
        for ax, (block_size, num_blocks) in zip(axes, montage.SETTINGS):
            with self.subTest(b=block_size, m=num_blocks):
                self.assertEqual(len(ax.images), 1)
                image = np.asarray(ax.images[0].get_array())
                self.assertEqual(image.shape, SHAPE)
                self.assertEqual(ax.images[0].get_clim(), (0, 1))
                # Every panel shows the same face: unoccluded pixels are unchanged.
                unchanged = image != 1.0
                np.testing.assert_array_equal(image[unchanged], clean[unchanged])
                self.assertEqual(int((image == 1.0).sum()) > 0, num_blocks > 0)
        self.assertEqual(axes[0].get_title(), "Clean (control)")
        self.assertEqual(axes[5].get_title(), "b=8, m=2")

    def test_pair_has_clean_occluded_and_mask_panels(self):
        V, labels = synthetic_data()
        fig, axes = montage.make_pair(V, labels, SHAPE)
        self.assertEqual([ax.get_title() for ax in axes],
                         ["Clean", "Occluded", "Occlusion mask"])

    def test_montage_helper_rejects_invalid_input(self):
        V = np.zeros((SHAPE[0] * SHAPE[1], 3))
        with self.assertRaisesRegex(ValueError, "one title per"):
            plot_occlusion_montage(V, SHAPE, ["a", "b"])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            plot_occlusion_montage(V + 2, SHAPE, ["a", "b", "c"])
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
