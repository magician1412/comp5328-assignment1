"""End-to-end check of main.py on a tiny synthetic ORL-style dataset.

Writes PGM files to a temporary folder, runs main.main() with two repeats and
checks the run folder, metadata.json and the CSV outputs. No supplied dataset
is needed.
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import main
from tests.test_data_loader import write_pgm

N_SUBJECTS, N_IMAGES, SIDE = 3, 4, 54  # reduce=3 gives 18 x 18 images (b=16 fits)


def write_dataset(root):
    """Write N_SUBJECTS ORL-style subject folders of random grey images."""
    rng = np.random.default_rng(0)
    for subject in range(1, N_SUBJECTS + 1):
        base = rng.integers(20, 200, size=(SIDE, SIDE))
        for image in range(1, N_IMAGES + 1):
            noise = rng.integers(0, 30, size=(SIDE, SIDE))
            write_pgm(root / "ORL" / f"s{subject}" / f"{image}.pgm", base + noise)


def read_csv(path):
    """Return the rows of a CSV file as dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class MainRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        write_dataset(root / "data")
        cls.out_dir = root / "results"
        main.main(["--dataset", "orl", "--repeats", "2",
                   "--data-root", str(root / "data"), "--out-dir", str(cls.out_dir)])
        cls.run_dir, = cls.out_dir.iterdir()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_run_folder_name(self):
        stamp, commit, dataset = self.run_dir.name.split("_")
        self.assertEqual(dataset, "orl")
        self.assertRegex(stamp, r"^\d{8}-\d{6}$")
        self.assertTrue(commit)

    def test_metadata_records_protocol(self):
        meta = json.loads((self.run_dir / "metadata.json").read_text())
        config = main.DATASET_CONFIGS["orl"]
        self.assertEqual(meta["rank"], N_SUBJECTS)
        self.assertEqual(meta["repeats"], 2)
        self.assertEqual(meta["sample_fraction"], main.SAMPLE_FRACTION)
        self.assertEqual([tuple(s) for s in meta["settings"]], main.SETTINGS)
        self.assertEqual(meta["seeds"]["sample"], main.SAMPLE_SEED)
        nmf, rnmf = meta["methods"]["Frobenius-NMF"], meta["methods"]["RNMF-L1"]
        self.assertEqual((nmf["tol"], nmf["max_iter"], nmf["min_iter"]),
                         (config["nmf"]["tol"], 2000, 200))
        self.assertEqual((rnmf["lambda"], rnmf["tol"], rnmf["max_iter"]),
                         (config["rnmf"]["lam"], 0.0, 2000))
        for key in ("commit", "commit_short", "dirty"):
            self.assertIn(key, meta["git"])
        for key in ("python", "numpy", "scipy", "scikit-learn"):
            self.assertTrue(meta["versions"][key])
        for key in ("started", "finished", "runtime_seconds"):
            self.assertIn(key, meta)

    def test_raw_csv_has_iterations_and_convergence(self):
        rows = read_csv(self.run_dir / "raw_results.csv")
        self.assertEqual(len(rows), 2 * len(main.SETTINGS) * 2)
        for row in rows:
            with self.subTest(method=row["algorithm"], b=row["block_size"],
                              m=row["num_blocks"], r=row["repeat"]):
                self.assertIn(row["converged"], ("True", "False"))
                iterations = int(row["iterations"])
                if row["algorithm"] == "RNMF-L1":
                    self.assertEqual(iterations, 2000)
                    self.assertEqual(row["converged"], "False")
                else:
                    self.assertTrue(200 <= iterations <= 2000)
                occluded = float(row["rre_occluded"])
                if row["num_blocks"] == "0":
                    self.assertTrue(np.isnan(occluded))
                else:
                    self.assertTrue(np.isfinite(occluded))
                self.assertTrue(np.isfinite(float(row["rre_unoccluded"])))

    def test_summary_has_paired_difference(self):
        rows = read_csv(self.run_dir / "summary.csv")
        self.assertEqual(len(rows), 2 * len(main.SETTINGS))
        for row in rows:
            self.assertEqual(row["paired_rre_diff_n"], "2")
            self.assertTrue(np.isfinite(float(row["paired_rre_diff_mean"])))
            self.assertTrue(np.isfinite(float(row["paired_rre_diff_std"])))

    def test_plots_written(self):
        names = {path.name for path in (self.run_dir / "plots").iterdir()}
        for name in ("orl_rre_sweeps.png", "orl_rre_sweeps.pdf", "orl_objective.png",
                     "orl_objective.pdf", "orl_summary_table.tex"):
            self.assertIn(name, names)

    def test_objective_trace_covers_repeat_0(self):
        rows = read_csv(self.run_dir / "objective_trace.csv")
        raw = [r for r in read_csv(self.run_dir / "raw_results.csv") if r["repeat"] == "0"]
        self.assertEqual(len(rows), sum(int(r["iterations"]) for r in raw))
        self.assertEqual(set(rows[0]), {"setting", "block_size", "num_blocks", "method",
                                        "iteration", "objective"})


if __name__ == "__main__":
    unittest.main()
