"""Load clean ORL / CroppedYaleB images with samples stored as columns.

Development example (with ``code/algorithm`` on ``sys.path``)::

    from utils.data_loader import load_data
    from utils.paths import DATA_DIR
    V_clean, labels, image_shape = load_data(DATA_DIR / "ORL", reduce=3)

For CroppedYaleB use ``load_data(DATA_DIR / "CroppedYaleB", reduce=4)``.
Pillow is confined to image loading; no factorization, noise, or evaluation is
performed here. Each call returns a fresh matrix. Future corruption code must
work on a copy of V_clean to preserve the clean reference.
"""

from numbers import Integral
from pathlib import Path
import re

import numpy as np
from PIL import Image


def load_data(root, reduce, dtype=np.float64):
    """Return ``(V_clean, labels, image_shape)`` for one dataset directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing ORL ``s<number>`` subjects or CroppedYaleB
        ``yaleB<two-digit number>`` subjects. Relative paths are resolved from
        the caller's working directory. Subject IDs need not be consecutive.
    reduce : int
        Positive integer downsampling factor. Width and height are floor-divided
        by this factor, using explicit Pillow BICUBIC interpolation.
    dtype : numpy floating dtype, optional
        Output floating dtype; defaults to float64. Integer dtypes are rejected.

    Returns
    -------
    V_clean : numpy.ndarray, shape (height * width, n_images)
        Grayscale pixels divided by 255, always in [0, 1]. Each column is a
        C-order (row-major) flattened image, reshaped with ``image_shape``.
    labels : numpy.ndarray, shape (n_images,)
        Contiguous int64 labels 0..C-1 for subjects with usable images. Subject
        directories are sorted by numeric subject ID (s1, s2, ..., s10).
        Filenames remain lexicographically sorted (1.pgm, 10.pgm, 2.pgm).
    image_shape : tuple[int, int]
        Resized (height, width), common to every returned image.

    Notes
    -----
    The .pgm extension and Ambient.pgm exclusion are case-insensitive. Files
    ending in .pgm.bad, metadata/log files, unrelated directories, and nested
    directories are ignored.
    Empty subjects are skipped without reserving a label. Counts are derived from
    disk; neither class counts nor sample counts are hard-coded. Mixing ORL and
    YaleB subject naming in a single root is rejected. Unreadable selected images,
    inconsistent resized dimensions, and datasets with no usable images raise
    informative errors rather than silently dropping face images.
    """
    if isinstance(reduce, (bool, np.bool_)) or not isinstance(reduce, Integral):
        raise ValueError("reduce must be a positive integer")
    if reduce < 1:
        raise ValueError("reduce must be a positive integer")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("dtype must be a real floating-point dtype")

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    subjects = sorted(
        (p for p in root.iterdir()
         if p.is_dir() and re.fullmatch(r"s[1-9][0-9]*|yaleB[0-9]{2}", p.name)),
        key=lambda p: int(p.name[5:] if p.name.startswith("yaleB") else p.name[1:]),
    )
    if len({p.name.startswith("yaleB") for p in subjects}) > 1:
        raise ValueError(f"Mixed ORL and YaleB subject directories in {root}")

    columns, labels = [], []
    image_shape = None
    next_label = 0
    for subject in subjects:
        paths = sorted(
            (p for p in subject.iterdir()
             if p.is_file() and p.suffix.lower() == ".pgm"
             and not p.name.lower().endswith("ambient.pgm")),
            key=lambda p: p.name,
        )
        if not paths:
            continue
        for path in paths:
            try:
                with Image.open(path) as source:
                    gray = source.convert("L")
                    width, height = (s // reduce for s in gray.size)
                    if width < 1 or height < 1:
                        raise ValueError(f"reduce={reduce} gives a zero-sized image")
                    resized = gray.resize(
                        (width, height), resample=Image.Resampling.BICUBIC
                    )
                    pixels = np.array(resized, dtype=dtype)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot load face image {path}: {exc}") from exc

            if image_shape is None:
                image_shape = pixels.shape
            elif pixels.shape != image_shape:
                raise ValueError(
                    f"Inconsistent resized shape for {path}: "
                    f"{pixels.shape}; expected {image_shape}"
                )
            pixels /= 255.0
            columns.append(pixels.reshape(-1, order="C"))
            labels.append(next_label)
        next_label += 1

    if not columns:
        raise ValueError(f"No usable .pgm face images in subject directories: {root}")
    return np.column_stack(columns), np.asarray(labels, dtype=np.int64), image_shape
