"""Independent Member 3 loader tests; no NMF or evaluation dependencies.

Run from the repository root (any directory works):
    pytest -q code/algorithm/tests/test_data_loader.py

The supplied-dataset checks use code/data/ located by tests/conftest.py and are
skipped when the datasets are absent.

Supplied-release counts belong only in these data checks, never in the loader.
Synthetic PGM fixtures use the standard library; Pillow remains in the loader.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from utils.data_loader import load_data


def write_pgm(path, pixels):
    """Write a tiny binary PGM fixture, without requiring Pillow in tests."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width = pixels.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels.tobytes())


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_order_labels_filtering_and_normalization(self):
        # Subject IDs sort numerically; filenames within a subject sort lexically.
        for subject, filename, value in [
            ("s2", "2.pgm", 255), ("s10", "1.pgm", 127),
            ("s1", "2.pgm", 100), ("s1", "10.pgm", 50),
            ("s1", "1.pgm", 0),
        ]:
            write_pgm(self.root / subject / filename, np.full((6, 8), value))
        for filename in [".DS_Store", "README", "s99"]:
            (self.root / filename).write_text("not an image")
        (self.root / "s3").mkdir()  # Empty subject must not leave a label gap.
        (self.root / "junk").mkdir()
        (self.root / "junk" / "1.pgm").write_text("not an image")
        for filename in ["face_Ambient.pgm", "face.pgm.bad",
                         "WS_FTP.LOG", "face.info", "DEADJOE", ".DS_Store"]:
            (self.root / "s1" / filename).write_text("not an image")
        (self.root / "s1" / "nested.pgm").mkdir()

        V, labels, shape = load_data(self.root, reduce=2)
        self.assertEqual(shape, (3, 4))
        self.assertEqual(V.shape, (12, 5))
        self.assertEqual(V.dtype, np.dtype(np.float64))
        self.assertEqual(labels.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 2])
        expected = np.broadcast_to(np.array([0, 50, 100, 255, 127]) / 255, V.shape)
        np.testing.assert_array_equal(V, expected)
        again, again_labels, again_shape = load_data(self.root, reduce=2)
        np.testing.assert_array_equal(again, V)
        np.testing.assert_array_equal(again_labels, labels)
        self.assertEqual(again_shape, shape)
        self.assertFalse(np.shares_memory(V, again))
        V[:] = 0
        self.assertEqual(again[0, 3], 1.0)

    def test_first_twelve_subjects_sort_numerically(self):
        for number in [12, 2, 11, 1, 10, 9, 8, 7, 6, 5, 4, 3]:
            write_pgm(self.root / f"s{number}" / "1.pgm", np.full((2, 2), number))
        V, labels, shape = load_data(self.root, reduce=1)
        self.assertEqual(shape, (2, 2))
        np.testing.assert_array_equal(V[0], np.arange(1, 13) / 255)
        np.testing.assert_array_equal(labels, np.arange(12))

    def test_pgm_extension_case_and_exclusions(self):
        for subject in ["s1", "yaleB01"]:
            with self.subTest(subject=subject):
                root = self.root / subject
                directory = root / subject
                for filename, value in [("3.PgM", 30), ("1.pgm", 10), ("2.PGM", 20)]:
                    write_pgm(directory / filename, np.full((2, 2), value))
                for filename in [
                    "face_Ambient.pgm", "face_AMBIENT.PGM", "face_ambient.PgM",
                    "face.pgm.bad", "face.PGM.BAD", "face.PgM.Bad",
                    "WS_FTP.LOG", "face.info", "DEADJOE", ".DS_Store",
                ]:
                    (directory / filename).write_text("not an image")
                V, labels, shape = load_data(root, reduce=1)
                self.assertEqual(V.shape, (4, 3))
                self.assertEqual(shape, (2, 2))
                np.testing.assert_array_equal(V[0], np.array([10, 20, 30]) / 255)
                np.testing.assert_array_equal(labels, [0, 0, 0])

    def test_flattening_and_optional_float32(self):
        pixels = np.arange(24, dtype=np.uint8).reshape(4, 6)
        write_pgm(self.root / "s1" / "1.pgm", pixels)
        V, labels, shape = load_data(self.root, reduce=1, dtype=np.float32)
        self.assertEqual(V.dtype, np.dtype(np.float32))
        self.assertEqual(shape, (4, 6))
        np.testing.assert_array_equal(V[:, 0].reshape(shape), pixels.astype(np.float32) / 255)
        np.testing.assert_array_equal(labels, [0])

    def test_yaleb_nonconsecutive_subjects_and_exclusions(self):
        for subject, value in [
            ("yaleB16", 160), ("yaleB12", 120), ("yaleB15", 150), ("yaleB13", 130),
        ]:
            directory = self.root / subject
            write_pgm(directory / f"{subject}_P00A+000E+00.pgm", np.full((9, 11), value))
            for suffix in ["_P00_Ambient.pgm", "_P00A+010E+00.pgm.bad", "_P00.info"]:
                (directory / (subject + suffix)).write_text("not an image")
        V, labels, shape = load_data(self.root, reduce=4)
        self.assertEqual(shape, (2, 2))  # Floor division on both dimensions.
        self.assertEqual(V.shape, (4, 4))
        np.testing.assert_array_equal(labels, [0, 1, 2, 3])
        np.testing.assert_array_equal(V[0], np.array([120, 130, 150, 160]) / 255)
        self.assertFalse((self.root / "yaleB14").exists())

    def test_invalid_arguments_and_empty_dataset(self):
        for factor in [0, -1, 1.5, True, np.bool_(True), "3"]:
            with self.subTest(reduce=factor), self.assertRaisesRegex(ValueError, "positive integer"):
                load_data(self.root, reduce=factor)
        for dtype in [np.uint8, np.int64, np.complex128, bool]:
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "floating-point"):
                load_data(self.root, reduce=1, dtype=dtype)
        with self.assertRaisesRegex(ValueError, "No usable"):
            load_data(self.root, reduce=1)
        with self.assertRaises(FileNotFoundError):
            load_data(self.root / "missing", reduce=1)
        (self.root / "file").write_text("not a directory")
        with self.assertRaises(NotADirectoryError):
            load_data(self.root / "file", reduce=1)

    def test_mixed_dataset_conventions_rejected(self):
        (self.root / "s1").mkdir()
        (self.root / "yaleB01").mkdir()
        with self.assertRaisesRegex(ValueError, "Mixed ORL and YaleB"):
            load_data(self.root, reduce=1)

    def test_bad_selected_image_has_path_in_error(self):
        directory = self.root / "s1"
        directory.mkdir()
        path = directory / "1.pgm"
        path.write_text("broken image")
        with self.assertRaises(ValueError) as error:
            load_data(self.root, reduce=1)
        self.assertIn(str(path), str(error.exception))

    def test_inconsistent_dimensions_and_excessive_reduction(self):
        write_pgm(self.root / "s1" / "1.pgm", np.zeros((4, 6)))
        write_pgm(self.root / "s1" / "2.pgm", np.zeros((6, 6)))
        with self.assertRaisesRegex(ValueError, "Inconsistent resized shape"):
            load_data(self.root, reduce=1)
        with self.assertRaisesRegex(ValueError, "zero-sized image"):
            load_data(self.root, reduce=7)


@pytest.mark.usefixtures("supplied_data")
class SuppliedDatasetTests(unittest.TestCase):
    def test_supplied_datasets(self):
        data_root = self.data_root
        for dataset, reduction, shape, count, classes in [
            ("ORL", 3, (37, 30), 400, 40),
            ("CroppedYaleB", 4, (48, 42), 2414, 38),
        ]:
            with self.subTest(dataset=dataset):
                root = data_root / dataset
                V, labels, image_shape = load_data(root, reduce=reduction)
                self.assertEqual(image_shape, shape)
                self.assertEqual(V.shape, (shape[0] * shape[1], count))
                self.assertEqual(labels.shape, (V.shape[1],))
                np.testing.assert_array_equal(np.unique(labels), np.arange(classes))
                self.assertEqual(V.dtype, np.dtype(np.float64))
                self.assertTrue(np.isfinite(V).all())
                self.assertTrue((V >= 0).all())
                self.assertTrue((V <= 1).all())
                second, second_labels, second_shape = load_data(root, reduce=reduction)
                np.testing.assert_array_equal(second, V)
                np.testing.assert_array_equal(second_labels, labels)
                self.assertEqual(second_shape, image_shape)
                ambient = list(root.glob("*/*Ambient.pgm"))
                bad = list(root.glob("*/*.pgm.bad"))
                if dataset == "CroppedYaleB":
                    self.assertEqual(len(ambient), 38)
                    self.assertEqual(len(bad), 18)
                    self.assertEqual(len(list(root.glob("*/*.pgm"))) - len(ambient), count)
                print(
                    f"\n{dataset}: V={V.shape}, labels={labels.shape}, "
                    f"image_shape={image_shape}, classes={classes}, "
                    f"dtype={V.dtype}, range=[{V.min():.12g}, {V.max():.12g}], "
                    f"ambient_excluded={len(ambient)}, bad_excluded={len(bad)}, "
                    "repeat_load_identical=True"
                )


if __name__ == "__main__":
    unittest.main()
