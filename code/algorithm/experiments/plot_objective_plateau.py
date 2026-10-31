"""Objective and RRE of RNMF-L1 vs iteration, showing the early plateau.

Reads experiments/lambda_study_results/rre_trajectory/<name>_trajectory.csv
and _rule_stops.csv (written by rre_trajectory.py); fits nothing. The figure
goes to code/figures/lambda_study/. Run from any directory:
    python -B code/algorithm/experiments/plot_objective_plateau.py
"""

import argparse
import csv
import sys
from pathlib import Path

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.paths import FIGURES_DIR  # noqa: E402

TRAJECTORY_DIR = Path(__file__).resolve().parent / "lambda_study_results" / "rre_trajectory"

COLORS = {"": "#52514e", "0.005": "#2a78d6", "0.01": "#eb6834", "0.02": "#1baf7a", "0.03": "#eda100"}
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    """Plot the objective plateau figure from the saved trajectory CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefix", default=str(TRAJECTORY_DIR / "orl_b8_m2"),
                        help="path prefix of the trajectory and rule-stop CSVs")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR / "lambda_study")
    args = parser.parse_args()
    with Path(f"{args.prefix}_trajectory.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["repeat"] == "0"]
    with Path(f"{args.prefix}_rule_stops.csv").open(newline="") as stream:
        stops = {r["lambda"]: r["rule_stop_iteration"] for r in csv.DictReader(stream)
                 if r["repeat"] == "0"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lam, color in COLORS.items():
        sel = sorted((r for r in rows if r["lambda"] == lam), key=lambda r: int(r["iteration"]))
        its = np.array([int(r["iteration"]) for r in sel])
        obj = np.array([float(r["objective"]) for r in sel])
        rre = np.array([float(r["rre"]) for r in sel])
        label = "Frobenius NMF" if lam == "" else f"RNMF-L1 λ={lam}"
        style = "--" if lam == "" else "-"
        axes[0].plot(its, obj / obj[-1], style, color=color, linewidth=1.8, label=label)
        axes[1].plot(its, rre, style, color=color, linewidth=1.8, label=label)
        stop = stops.get(lam)
        if stop:
            s = int(stop)
            for ax, y in ((axes[0], np.interp(s, its, obj / obj[-1])), (axes[1], np.interp(s, its, rre))):
                ax.plot([s], [y], marker="v", color=color, markersize=8,
                        markeredgecolor="white", linestyle="none")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Objective / objective at iteration 5000")
    axes[0].set_title("Objective (training loss on occluded V)", loc="left", fontsize=10)
    axes[1].set_ylabel("RRE against clean data")
    axes[1].set_title("RRE against clean data", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_xlim(0, 3000)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False, fontsize=8.5)
    fig.text(0.5, -0.09, "ORL, b=8, m=2, k=40, repeat 0 (tuning seeds), no early stopping, values every "
             "50 iterations. ▼ = where the window-5 / min_iter-200 rule (tol 1e-3) would stop.",
             ha="center", va="top", fontsize=8, color="#52514e")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure = args.figure_dir / f"{Path(args.prefix).name}_objective_plateau"
    for ext in ("png", "pdf"):
        fig.savefig(f"{figure}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {figure}.png")


if __name__ == "__main__":
    main()
