"""Summaries, report figures and the lambda choice for the RNMF-L1 study.

Reads only files written by rnmf_lambda_sensitivity.py, so it never refits.
Run from any directory:
    python -B code/algorithm/experiments/plot_lambda_sensitivity.py

Selection rule (fixed before the results were seen), per dataset, using only
the tuning setting b=8, m=2 and RNMF-L1 at the fixed budget:
  1. take the lambda with the lowest mean RRE over repeats;
  2. every lambda whose mean RRE is within one sample standard deviation
     (ddof=1) of that best
     mean counts as tied; choose the LARGEST tied lambda (sparser S, smaller
     loss on unoccluded data);
  3. flag the result if the best lambda lies on the edge of the grid.

Outputs: <dataset>_summary.csv and lambda_selection.json next to the raw
CSVs (experiments/lambda_study_results/lambda_sensitivity/); lambda_summary.md
and the figures lambda_sensitivity_rre, lambda_sensitivity_occluded and
lambda_sensitivity_S (PNG and PDF) in code/figures/lambda_study/. The face
figure <dataset>_lambda_reconstructions is drawn only when the gitignored
<dataset>_examples_b8_m2.npz written by rnmf_lambda_sensitivity.py exists.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.paths import FIGURES_DIR  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "lambda_study_results" / "lambda_sensitivity"
TUNING = (8, 2)
CLEAN = (4, 0)
NAMES = {"orl": "ORL", "yaleb": "Extended YaleB"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S_CMAP = LinearSegmentedColormap.from_list("signed", ["#2a78d6", "#f0efec", "#e34948"])
METRICS = ["rre", "rre_occluded", "rre_unoccluded", "s_nonzero", "s_mass_in_mask",
           "mask_recall", "iterations", "runtime_seconds"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _float(value):
    """Convert a CSV cell to float; empty cells become NaN."""
    return float(value) if value not in ("", None) else float("nan")


def summarize(raw_path):
    """Group raw fits by (setting, algorithm, lambda); return summary rows."""
    with Path(raw_path).open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    groups = {}
    for row in raw:
        key = (int(row["block_size"]), int(row["num_blocks"]), row["algorithm"], row["lambda"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (b, m, algorithm, lam), rows in sorted(groups.items(), key=lambda kv: (
            -kv[0][1], kv[0][0], kv[0][2] != "Frobenius-NMF", _float(kv[0][3]))):
        out = {"dataset": rows[0]["dataset"], "block_size": b, "num_blocks": m,
               "algorithm": algorithm, "lambda": lam, "n_runs": len(rows),
               "max_iter": rows[0]["max_iter"], "tol": rows[0]["tol"],
               "converged_runs": sum(r["converged"] == "True" for r in rows),
               "iterations_max": max(int(r["iterations"]) for r in rows)}
        for metric in METRICS:
            values = np.array([_float(r[metric]) for r in rows])
            out[f"{metric}_mean"] = float(np.mean(values))
            # Sample standard deviation over repeats (ddof=1).
            out[f"{metric}_std"] = (float(np.std(values, ddof=1)) if values.size > 1
                                    else float("nan"))
        summary.append(out)
    return summary


def _rows(summary, setting, algorithm="RNMF-L1"):
    """Return the summary rows of one method on one (b, m) setting."""
    return [r for r in summary if (r["block_size"], r["num_blocks"]) == setting
            and r["algorithm"] == algorithm]


def select_lambda(summary):
    """Apply the pre-specified rule on the tuning setting."""
    rows = _rows(summary, TUNING)
    grid = sorted(_float(r["lambda"]) for r in rows)
    best = min(rows, key=lambda r: r["rre_mean"])
    threshold = best["rre_mean"] + best["rre_std"]
    tied = sorted(_float(r["lambda"]) for r in rows if r["rre_mean"] <= threshold)
    chosen = max(tied)
    chosen_row = next(r for r in rows if _float(r["lambda"]) == chosen)
    nmf = _rows(summary, TUNING, "Frobenius-NMF")[0]
    return {
        "grid": grid, "n_repeats": best["n_runs"],
        "best_lambda": _float(best["lambda"]),
        "best_rre": [best["rre_mean"], best["rre_std"]],
        "tie_threshold": threshold, "tied_lambdas": tied,
        "chosen_lambda": chosen,
        "chosen_rre": [chosen_row["rre_mean"], chosen_row["rre_std"]],
        "frobenius_nmf_rre": [nmf["rre_mean"], nmf["rre_std"]],
        "best_on_grid_edge": _float(best["lambda"]) in (grid[0], grid[-1]),
    }


def _series(summary, setting, key):
    """Return (lambda, mean, std) arrays of one metric for RNMF-L1."""
    rows = _rows(summary, setting)
    lam = np.array([_float(r["lambda"]) for r in rows])
    return (lam, np.array([r[f"{key}_mean"] for r in rows]),
            np.array([r[f"{key}_std"] for r in rows]))


def _nmf(summary, setting, key="rre"):
    """Return the Frobenius NMF mean of one metric (None if absent)."""
    rows = _rows(summary, setting, "Frobenius-NMF")
    return rows[0][f"{key}_mean"] if rows else None


def _style(ax, title):
    """Apply the shared log-lambda axis style and title."""
    ax.set_xscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel(r"$\lambda$ (log scale)")
    ax.set_title(title, fontsize=10, loc="left", color=INK)


def _line(ax, lam, mu, sd, color, label, marker="o"):
    """Draw a mean line with a ± std band."""
    ax.plot(lam, mu, color=color, linewidth=2, marker=marker, markersize=4.5, label=label)
    ax.fill_between(lam, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)


def _mark_choice(ax, selection):
    """Dotted line at the chosen lambda; its value goes into the panel title."""
    ax.axvline(selection["chosen_lambda"], color=INK2, linewidth=1, linestyle=":",
               label="λ chosen on b=8, m=2")


def _save(fig, out, name):
    """Save a figure as PNG and PDF and close it."""
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _finish(fig, axes, data, ncol=2):
    """Shared legend below the panels, then the repeat/seed note."""
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False, fontsize=8)
    reps = []
    for dataset, (summary, _) in data.items():
        counts = "/".join(str(_rows(summary, s)[0]["n_runs"]) for s in (TUNING, CLEAN)
                          if _rows(summary, s))
        reps.append(f"{NAMES[dataset]} {counts}")
    fig.text(0.5, -0.17, "Mean ± std over repeats (occluded/no occlusion: " + ", ".join(reps)
             + "); tuning seeds disjoint from the evaluation seeds.\nRNMF-L1: fixed 2000 "
             "iterations. Frobenius NMF (dashed): tolerance stopping.",
             fontsize=8, color=INK2, ha="center", va="top")


def plot_rre(data, out):
    """Overall RRE vs lambda for the tuning and no-occlusion settings."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.8), squeeze=False)
    for ax, (dataset, (summary, selection)) in zip(axes[0], data.items()):
        for setting, color, desc in ((TUNING, BLUE, "occluded, b=8, m=2"),
                                     (CLEAN, ORANGE, "no occlusion")):
            if not _rows(summary, setting):
                continue
            _line(ax, *_series(summary, setting, "rre"), color, f"RNMF-L1, {desc}")
            ref = _nmf(summary, setting)
            if ref is not None:
                ax.axhline(ref, color=color, linewidth=1.2, linestyle="--",
                           label=f"Frobenius NMF, {desc}")
        _mark_choice(ax, selection)
        _style(ax, f"{NAMES[dataset]} (chosen λ = {selection['chosen_lambda']:g})")
        ax.set_ylabel("RRE against clean data")
    _finish(fig, axes, data)
    _save(fig, out, "lambda_sensitivity_rre")


def plot_occluded(data, out):
    """Occluded vs unoccluded RRE on the tuning setting: the lambda trade-off."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.8), squeeze=False)
    for ax, (dataset, (summary, selection)) in zip(axes[0], data.items()):
        for key, color, desc in (("rre_occluded", ORANGE, "occluded pixels"),
                                 ("rre_unoccluded", AQUA, "unoccluded pixels")):
            _line(ax, *_series(summary, TUNING, key), color, f"RNMF-L1, {desc}")
            ax.axhline(_nmf(summary, TUNING, key), color=color, linewidth=1.2, linestyle="--",
                       label=f"Frobenius NMF, {desc}")
        _mark_choice(ax, selection)
        _style(ax, f"{NAMES[dataset]}, b=8, m=2 (chosen λ = {selection['chosen_lambda']:g})")
        ax.set_ylabel("RRE restricted to pixel group")
    _finish(fig, axes, data)
    _save(fig, out, "lambda_sensitivity_occluded")


def plot_s(data, out):
    """Sparsity of S and how much of it lies on the occlusion."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.6), squeeze=False)
    for ax, (dataset, (summary, selection)) in zip(axes[0], data.items()):
        for key, color, label in (("s_nonzero", BLUE, "Entries with S ≠ 0"),
                                  ("mask_recall", ORANGE, "Occluded pixels with S ≠ 0"),
                                  ("s_mass_in_mask", AQUA, "Share of |S| inside the occlusion")):
            _line(ax, *_series(summary, TUNING, key), color, label)
        _mark_choice(ax, selection)
        _style(ax, f"{NAMES[dataset]}, b=8, m=2 (chosen λ = {selection['chosen_lambda']:g})")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction")
    _finish(fig, axes, data, ncol=4)
    _save(fig, out, "lambda_sensitivity_S")


def plot_reconstructions(dataset, selection, examples_dir, out):
    """Clean / occluded / NMF W H / RNMF W H and S for three lambdas."""
    path = examples_dir / f"{dataset}_examples_b{TUNING[0]}_m{TUNING[1]}.npz"
    if not path.exists():
        print(f"{dataset}: {path.name} not found, reconstruction figure skipped")
        return
    data = np.load(path)
    h, w = (int(v) for v in data["image_shape"])
    grid = selection["grid"]
    show = sorted({grid[0], selection["chosen_lambda"], grid[-1]})
    panels = [("Clean\n", data["clean"], "img"), ("Occluded\ninput", data["noisy"], "img"),
              ("Frobenius\nNMF WH", data["WH_nmf"], "img")]
    for lam in show:
        star = "*" if lam == selection["chosen_lambda"] else ""
        panels += [(f"RNMF WH\nλ={lam:g}{star}", data[f"WH_lam_{lam:g}"], "img"),
                   (f"RNMF S\nλ={lam:g}{star}", data[f"S_lam_{lam:g}"], "S")]
    n_img = data["clean"].shape[1]
    fig, axes = plt.subplots(n_img, len(panels), figsize=(1.3 * len(panels), 1.5 * n_img))
    image = None
    for i in range(n_img):
        for c, (title, M, kind) in enumerate(panels):
            ax = axes[i, c]
            pic = M[:, i].reshape(h, w)
            if kind == "S":
                image = ax.imshow(pic, cmap=S_CMAP, vmin=-1, vmax=1)
            else:
                ax.imshow(np.clip(pic, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=7.5, color=INK)
    bar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    bar.set_label("S (red: brighter than WH)", fontsize=7, color=INK2)
    fig.suptitle(f"{NAMES[dataset]}, b=8, m=2, repeat 0; * chosen λ (WH shown clipped to [0, 1])",
                 fontsize=8, color=INK2, y=0.995)
    _save(fig, out, f"{dataset}_lambda_reconstructions")


def write_markdown(data, out):
    """Plain tables of every summary row, for the README and the report."""
    lines = ["# Lambda sensitivity summary", ""]
    for dataset, (summary, sel) in data.items():
        lines += [f"## {NAMES[dataset]}", "",
                  f"Chosen λ = **{sel['chosen_lambda']:g}** (best λ = {sel['best_lambda']:g}, "
                  f"RRE {sel['best_rre'][0]:.4f} ± {sel['best_rre'][1]:.4f}; tied: "
                  f"{', '.join(f'{v:g}' for v in sel['tied_lambdas'])}"
                  + ("; best λ on grid edge" if sel["best_on_grid_edge"] else "") + ").",
                  ""]
        for setting in (TUNING, CLEAN):
            rows = [r for r in summary if (r["block_size"], r["num_blocks"]) == setting]
            if not rows:
                continue
            lines += [f"### b={setting[0]}, m={setting[1]} ({rows[0]['n_runs']} repeats)", "",
                      "| Method | λ | RRE | RRE occluded | RRE unoccluded | S ≠ 0 | S mass in mask | iterations | s/fit |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                def f(key, digits=3):
                    """Format one metric as mean ± std."""
                    mu = r[f"{key}_mean"]
                    return "–" if np.isnan(mu) else f"{mu:.{digits}f} ± {r[f'{key}_std']:.{digits}f}"
                lines.append(f"| {r['algorithm']} | {r['lambda'] or '–'} | {f('rre')} | "
                             f"{f('rre_occluded')} | {f('rre_unoccluded')} | {f('s_nonzero', 2)} | "
                             f"{f('s_mass_in_mask', 2)} | {r['iterations_mean']:.0f} | "
                             f"{r['runtime_seconds_mean']:.1f} |")
            lines.append("")
    (out / "lambda_summary.md").write_text("\n".join(lines))


def main():
    """Summarise the raw CSVs, select lambda and draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR,
                        help="folder with the raw CSVs; summaries are written here")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR / "lambda_study")
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for dataset in ("orl", "yaleb"):
        raw = args.dir / f"{dataset}_raw.csv"
        if not raw.exists():
            continue
        summary = summarize(raw)
        with (args.dir / f"{dataset}_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        data[dataset] = (summary, select_lambda(summary))
    plot_rre(data, args.figure_dir)
    plot_occluded(data, args.figure_dir)
    plot_s(data, args.figure_dir)
    for dataset, (_, selection) in data.items():
        plot_reconstructions(dataset, selection, args.dir, args.figure_dir)
    write_markdown(data, args.figure_dir)
    selections = {d: s for d, (_, s) in data.items()}
    (args.dir / "lambda_selection.json").write_text(json.dumps(selections, indent=2))
    for dataset, sel in selections.items():
        print(f"{dataset}: chosen λ={sel['chosen_lambda']:g}, best λ={sel['best_lambda']:g} "
              f"(RRE {sel['best_rre'][0]:.4f} ± {sel['best_rre'][1]:.4f}), tied {sel['tied_lambdas']}, "
              f"NMF {sel['frobenius_nmf_rre'][0]:.4f}" + ("  [edge]" if sel["best_on_grid_edge"] else ""))


if __name__ == "__main__":
    main()
