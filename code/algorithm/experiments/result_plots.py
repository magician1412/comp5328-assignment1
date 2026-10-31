"""Result figures and a LaTeX table built from the CSV files of one run.

Every function reads saved results only (summary.csv, objective_trace.csv);
nothing is refitted and no face images are drawn.
"""

from pathlib import Path
import csv

import numpy as np

METHODS = ("Frobenius-NMF", "RNMF-L1")
STYLE = {
    "Frobenius-NMF": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "RNMF-L1": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
}
DATASET_TITLES = {"ORL": "ORL", "CroppedYaleB": "Extended YaleB"}
# Controlled sweeps of the main experiment.
FIXED_M = 1   # RRE vs block size b
FIXED_B = 8   # RRE vs block count m (m = 0 is the no-occlusion setting)


def read_rows(path):
    """Read a CSV file into a list of dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(value):
    """Convert a CSV cell to float; empty cells become NaN."""
    return float(value) if value not in ("", None) else float("nan")


def sweep_series(summary, method, x_field, select):
    """Return (x, mean, std) arrays of RRE for one method, sorted by x."""
    rows = [r for r in summary if r["algorithm"] == method and select(r)]
    rows.sort(key=lambda r: int(r[x_field]))
    x = np.array([int(r[x_field]) for r in rows])
    mean = np.array([_float(r["rre_mean"]) for r in rows])
    std = np.array([_float(r["rre_std"]) for r in rows])
    return x, mean, std


def _errorbar(ax, x, mean, std, method):
    """Draw one mean +/- std series; NaN stds (single repeat) draw no bar."""
    yerr = np.where(np.isfinite(std), std, 0.0) if np.isfinite(std).any() else None
    ax.errorbar(x, mean, yerr=yerr, capsize=3, linewidth=1.8, markersize=5,
                label=method, **STYLE[method])


def plot_rre_sweeps(summary, output_stem):
    """Save RRE vs b (m = FIXED_M) and RRE vs m (b = FIXED_B) side by side.

    The m panel includes the no-occlusion setting as m = 0 (its block size
    is irrelevant). Writes ``output_stem`` + .png and .pdf; returns the paths.
    """
    import matplotlib.pyplot as plt

    dataset = summary[0]["dataset"]
    repeats = summary[0]["repeats"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    panels = [
        (axes[0], "block_size", lambda r: int(r["num_blocks"]) == FIXED_M,
         "Block edge length $b$", f"Block size sweep ($m = {FIXED_M}$)"),
        (axes[1], "num_blocks",
         lambda r: int(r["block_size"]) == FIXED_B or int(r["num_blocks"]) == 0,
         "Number of blocks $m$ ($m = 0$: no occlusion)",
         f"Block count sweep ($b = {FIXED_B}$)"),
    ]
    for ax, x_field, select, xlabel, title in panels:
        for method in METHODS:
            x, mean, std = sweep_series(summary, method, x_field, select)
            if x.size:
                _errorbar(ax, x, mean, std, method)
                ax.set_xticks(x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RRE against clean data")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{DATASET_TITLES.get(dataset, dataset)}: RRE, mean ± std over "
                 f"{repeats} repeat(s)", fontsize=11)
    return _save(fig, output_stem)


def plot_objective_traces(trace_rows, output_stem, dataset=""):
    """Save objective vs iteration (repeat 0), one panel per method.

    Each panel has one line per occlusion setting; the y axis is logarithmic.
    Writes ``output_stem`` + .png and .pdf; returns the paths.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METHODS), figsize=(10, 3.8), layout="constrained")
    settings = list(dict.fromkeys(r["setting"] for r in trace_rows))
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, max(len(settings), 1)))
    for ax, method in zip(axes, METHODS):
        for setting, color in zip(settings, colors):
            rows = [r for r in trace_rows
                    if r["method"] == method and r["setting"] == setting]
            if not rows:
                continue
            iterations = np.array([int(r["iteration"]) for r in rows])
            objective = np.array([float(r["objective"]) for r in rows])
            label = "no occlusion" if setting.endswith(":0") else f"b:m = {setting}"
            ax.plot(iterations, objective, color=color, linewidth=1.4, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective (log scale)")
        ax.set_title(method, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{DATASET_TITLES.get(dataset, dataset)}: objective vs iteration "
                 "(repeat 0)", fontsize=11)
    return _save(fig, output_stem)


def _save(fig, output_stem):
    """Save a figure as PNG and PDF, close it and return both paths."""
    import matplotlib.pyplot as plt

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_latex_table(rows, output_path):
    """Write an RRE / paired difference / Accuracy / NMI table (mean ± std)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def cell(row, metric):
        """Format one metric as a LaTeX mean ± std cell."""
        return (f"{_float(row[f'{metric}_mean']):.4f} $\\pm$ "
                f"{_float(row[f'{metric}_std']):.4f}")

    lines = [
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Dataset & Algorithm & $b$ & $m$ & RRE & $\Delta$RRE (NMF$-$RNMF) "
        r"& Accuracy & NMI \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = row["dataset"].replace("_", r"\_")
        algorithm = row["algorithm"].replace("_", r"\_")
        lines.append(
            f"{dataset} & {algorithm} & {row['block_size']} & {row['num_blocks']} & "
            f"{cell(row, 'rre')} & {cell(row, 'paired_rre_diff')} & "
            f"{cell(row, 'accuracy')} & {cell(row, 'nmi')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
