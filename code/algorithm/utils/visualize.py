"""Member 3 Helpers for visualizing clean, occluded, and reconstructed images."""

from numbers import Integral
from pathlib import Path

import numpy as np


def column_to_image(V, image_shape, sample_index):
    """Reconstruct one column as an independent C-order (height, width) array.

    V must be 2D with samples as columns. image_shape must contain exactly two
    positive integer dimensions whose product equals V.shape[0]. sample_index
    must be a Python/NumPy integer in [0, V.shape[1]); booleans are rejected for
    both dimensions and the index. Dtype and values are preserved, including
    boolean masks. This reshape helper does not normalize or validate intensity
    ranges. The returned array has independent storage so edits cannot alter V.
    """
    V = np.asarray(V)
    if V.ndim != 2:
        raise ValueError("V must be a 2D matrix with samples as columns")
    try:
        dimensions = tuple(image_shape)
    except TypeError as exc:
        raise TypeError("image_shape must contain two integer dimensions") from exc
    if len(dimensions) != 2:
        raise ValueError("image_shape must contain exactly (height, width)")
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("image dimensions must be integers, not booleans")
        if dimension <= 0:
            raise ValueError("image dimensions must be positive")
    height, width = map(int, dimensions)
    if height * width != V.shape[0]:
        raise ValueError("height * width must match the number of V rows")
    if isinstance(sample_index, (bool, np.bool_)) or not isinstance(sample_index, Integral):
        raise TypeError("sample_index must be an integer, not a boolean")
    if not 0 <= sample_index < V.shape[1]:
        raise ValueError("sample_index must satisfy 0 <= sample_index < V.shape[1]")
    return V[:, int(sample_index)].reshape((height, width), order="C").copy()


def plot_occlusion_pair(
    V_clean, V_noisy, image_shape, sample_index,
    mask=None, title=None, save_path=None,
):
    """Return (fig, axes) showing Clean, Occluded, and optionally Occlusion mask.

    Clean/noisy inputs must be equally shaped 2D real numeric matrices containing
    only finite values in [0, 1]. If supplied, mask must have the same shape and
    boolean dtype. All panels use the same C-order reconstruction, grayscale
    limits [0, 1], an upper origin and nearest-neighbor display interpolation.
    No inputs are modified; this function does not generate noise or run models.

    axes is always a 1D array of two or three Matplotlib Axes. title, if supplied,
    becomes the figure-level title. The function never calls show(); callers own
    the returned Figure and should close it when finished. The caller chooses
    the Matplotlib backend (use Agg for headless execution).

    With save_path=None, no figure file is written. Otherwise, parent directories
    are created and the figure is saved to exactly that path with a tight bounding
    box. An existing file at that explicit path is overwritten. The suffix selects
    the format; a path without a suffix receives PNG content without renaming.
    """
    V_clean, V_noisy = np.asarray(V_clean), np.asarray(V_noisy)
    for name, V in [("V_clean", V_clean), ("V_noisy", V_noisy)]:
        if V.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        if V.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(V).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(V < 0) or np.any(V > 1):
            raise ValueError(f"{name} values must lie in [0, 1]")
    if V_clean.shape != V_noisy.shape:
        raise ValueError("V_clean and V_noisy must have the same shape")

    clean_image = column_to_image(V_clean, image_shape, sample_index)
    # Reuse the validated dimensions, including for single-pass shape iterables.
    shape = clean_image.shape
    images = [clean_image, column_to_image(V_noisy, shape, sample_index)]
    panel_titles = ["Clean", "Occluded"]
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != V_clean.shape:
            raise ValueError("mask must have the same shape as V_clean")
        if mask.dtype != np.dtype(bool):
            raise TypeError("mask must have boolean dtype")
        images.append(column_to_image(mask, shape, sample_index))
        panel_titles.append("Occlusion mask")

    # Keep Matplotlib out of loading/sampling/occlusion and reshape-only calls.
    import matplotlib.pyplot as plt

    fig, axes_grid = plt.subplots(
        1, len(images), figsize=(3 * len(images), 3.5), squeeze=False,
        layout="constrained",
    )
    axes = axes_grid[0]
    try:
        for ax, image, panel_title in zip(axes, images, panel_titles):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1,
                      origin="upper", interpolation="nearest")
            ax.set_title(panel_title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
        if title is not None:
            fig.suptitle(title)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, format=path.suffix.lstrip(".").lower() or "png",
                        dpi=150, bbox_inches="tight")
    except Exception:
        plt.close(fig)
        raise
    return fig, axes


def plot_occlusion_montage(
    V_panels, image_shape, panel_titles,
    title=None, save_path=None, width=6.3, title_fontsize=7,
):
    """Return (fig, axes) showing every column of V_panels side by side in one row.

    V_panels is a 2D real numeric matrix of shape (d, p) containing only finite
    values in [0, 1]; column j is a C-order flattened image shown in panel j,
    left to right. panel_titles must contain exactly p strings. All panels use
    fixed grayscale limits [0, 1] (so occlusion = 1.0 is always true white), an
    upper origin and nearest-neighbor display interpolation. No inputs are
    modified; this function does not generate noise or run models.

    width is the figure width in inches; the default 6.3 in (16 cm) suits a
    LaTeX \\textwidth figure, and the height follows from the image aspect ratio.
    axes is always a 1D array of p Matplotlib Axes. title and save_path behave
    as in plot_occlusion_pair: the function never calls show(), callers own the
    returned Figure, and the save_path suffix selects the format.
    """
    V_panels = np.asarray(V_panels)
    if V_panels.ndim != 2:
        raise ValueError("V_panels must be a 2D matrix with panels as columns")
    if V_panels.dtype.kind not in "iuf":
        raise TypeError("V_panels must contain real numeric values")
    if not np.isfinite(V_panels).all():
        raise ValueError("V_panels must contain only finite values")
    if np.any(V_panels < 0) or np.any(V_panels > 1):
        raise ValueError("V_panels values must lie in [0, 1]")
    panel_titles = [str(panel_title) for panel_title in panel_titles]
    n_panels = V_panels.shape[1]
    if n_panels < 1 or len(panel_titles) != n_panels:
        raise ValueError("panel_titles must give one title per V_panels column")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be a positive number of inches")

    first = column_to_image(V_panels, image_shape, 0)
    # Reuse the validated dimensions, including for single-pass shape iterables.
    shape = first.shape
    images = [first] + [column_to_image(V_panels, shape, j) for j in range(1, n_panels)]

    import matplotlib.pyplot as plt

    # Height: one image row at the panel aspect ratio plus room for the titles.
    panel_width = width / n_panels
    height = panel_width * shape[0] / shape[1] + 0.3 + (0.25 if title else 0.0)
    fig, axes_grid = plt.subplots(
        1, n_panels, figsize=(width, height), squeeze=False,
        layout="constrained",
    )
    axes = axes_grid[0]
    try:
        for ax, image, panel_title in zip(axes, images, panel_titles):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1,
                      origin="upper", interpolation="nearest")
            ax.set_title(panel_title, fontsize=title_fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            # Thin frame: white blocks at an image edge stay visible on a white page.
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_edgecolor("0.5")
        if title is not None:
            fig.suptitle(title, fontsize=title_fontsize + 1)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, format=path.suffix.lstrip(".").lower() or "png",
                        dpi=300, bbox_inches="tight")
    except Exception:
        plt.close(fig)
        raise
    return fig, axes


def plot_reconstruction_comparison(
    V_clean,
    V_occluded,
    V_baseline_hat,
    V_rnmf_hat,
    image_shape,
    sample_index,
    title=None,
    save_path=None,
    baseline_name="Baseline NMF",
    rnmf_name="RNMF",
):
    """Return (fig, axes) showing Clean, Occluded, Baseline, and RNMF images.

    Parameters
    ----------
    V_clean : array-like
        Clean data matrix with samples stored as columns.

    V_occluded : array-like
        Occluded/noisy data matrix with samples stored as columns.

    V_baseline_hat : array-like
        Reconstructed matrix from the baseline NMF method.

    V_rnmf_hat : array-like
        Reconstructed matrix from the RNMF method.

    image_shape : tuple[int, int]
        Image height and width.

    sample_index : int
        Which column/sample to visualize.

    title : str, optional
        Figure-level title.

    save_path : path-like, optional
        If supplied, save the figure to this path.

    baseline_name : str, optional
        Title for the baseline reconstruction panel.

    rnmf_name : str, optional
        Title for the RNMF reconstruction panel.
    """
    V_clean = np.asarray(V_clean)
    V_occluded = np.asarray(V_occluded)
    V_baseline_hat = np.asarray(V_baseline_hat)
    V_rnmf_hat = np.asarray(V_rnmf_hat)

    # Validate clean and occluded matrices
    for name, V in [("V_clean", V_clean), ("V_occluded", V_occluded)]:
        if V.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        if V.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(V).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(V < 0) or np.any(V > 1):
            raise ValueError(f"{name} values must lie in [0, 1]")

    # Validate reconstruction matrices
    for name, V in [("V_baseline_hat", V_baseline_hat), ("V_rnmf_hat", V_rnmf_hat)]:
        if V.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        if V.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(V).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(V < 0):
            raise ValueError(f"{name} must be non-negative")

    if not (
        V_clean.shape
        == V_occluded.shape
        == V_baseline_hat.shape
        == V_rnmf_hat.shape
    ):
        raise ValueError(
            "V_clean, V_occluded, V_baseline_hat, and V_rnmf_hat must have the same shape"
        )

    clean_image = column_to_image(V_clean, image_shape, sample_index)
    shape = clean_image.shape

    occluded_image = column_to_image(V_occluded, shape, sample_index)
    baseline_image = column_to_image(V_baseline_hat, shape, sample_index)
    rnmf_image = column_to_image(V_rnmf_hat, shape, sample_index)

    # Reconstruction values may slightly exceed 1; clip only for display.
    baseline_display = np.clip(baseline_image, 0.0, 1.0)
    rnmf_display = np.clip(rnmf_image, 0.0, 1.0)

    images = [
        clean_image,
        occluded_image,
        baseline_display,
        rnmf_display,
    ]

    panel_titles = [
        "Clean",
        "Occluded",
        f"{baseline_name} Reconstruction",
        f"{rnmf_name} Reconstruction",
    ]

    import matplotlib.pyplot as plt

    fig, axes_grid = plt.subplots(
        1, 4,
        figsize=(12, 3.5),
        squeeze=False,
        layout="constrained",
    )
    axes = axes_grid[0]

    try:
        for ax, image, panel_title in zip(axes, images, panel_titles):
            ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(panel_title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

        if title is not None:
            fig.suptitle(title)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path,
                format=path.suffix.lstrip(".").lower() or "png",
                dpi=150,
                bbox_inches="tight",
            )

    except Exception:
        plt.close(fig)
        raise

    return fig, axes