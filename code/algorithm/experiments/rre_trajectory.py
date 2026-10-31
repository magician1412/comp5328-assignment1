"""RRE and objective as a function of iteration, without early stopping.

Used to choose the stopping protocol: it shows whether RRE against the clean
data settles, and where the window-5 / min_iter-200 rule would have stopped.

Each fit is run in chunks of --step iterations with tol=0, warm-starting from
the previous W, H. This is exactly equivalent to one long run: RNMF-L1
recomputes S from the current W, H every iteration and the baseline keeps no
other state.

Run from any directory (the repository root is shown):
    python -B code/algorithm/experiments/rre_trajectory.py --dataset orl
    python -B code/algorithm/experiments/rre_trajectory.py --plot-only --dataset orl

CSV outputs go to experiments/lambda_study_results/rre_trajectory/ and the
figure to code/figures/lambda_study/.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

import numpy as np  # noqa: E402

from experiments.experiment import _shared_initialization  # noqa: E402
from nmf.nmf_baseline import nmf_baseline  # noqa: E402
from nmf.nmf_rnmf_l1 import fit_rnmf_l1  # noqa: E402
from utils.data_loader import load_data  # noqa: E402
from utils.evaluation_metrics import (  # noqa: E402
    masked_relative_reconstruction_error,
    relative_reconstruction_error,
)
from utils.occlusion import apply_occlusion  # noqa: E402
from utils.paths import DATA_DIR, FIGURES_DIR  # noqa: E402
from utils.sampling import sample_columns  # noqa: E402

DATASETS = {
    "orl": {"folder": "ORL", "reduce": 3, "tol": 1e-3},
    "yaleb": {"folder": "CroppedYaleB", "reduce": 4, "tol": 5e-4},
}
OUTPUT_DIR = Path(__file__).resolve().parent / "lambda_study_results" / "rre_trajectory"
FIGURE_DIR = FIGURES_DIR / "lambda_study"
SAMPLE_SEED, CORRUPTION_SEED, INIT_SEED = 1000, 2000, 3000  # tuning seeds
FIELDS = ["dataset", "block_size", "num_blocks", "repeat", "method", "lambda",
          "iteration", "objective", "rre", "rre_occluded", "rre_unoccluded",
          "elapsed_seconds"]


def _name(args, b, m):
    """Output file prefix; a rank suffix is added only when --rank is given."""
    suffix = f"_k{args.rank}" if args.rank else ""
    return f"{args.dataset}_b{b}_m{m}{suffix}"


def _base(args, b, m):
    """Path prefix of the CSV outputs."""
    return args.output_dir / _name(args, b, m)


def window_stop(objective, tol, window=5, min_iter=200):
    """Iteration at which main.py's stopping rule would fire (None if never)."""
    obj = np.asarray(objective)
    for i in range(max(min_iter, window + 1), len(obj) + 1):
        recent = obj[i - window - 1:i]
        if np.abs(np.diff(recent)).sum() / (abs(recent[0]) + 1e-10) < tol:
            return i
    return None


def _rres(V_clean, W, H, mask):
    """Return (RRE, RRE on occluded pixels, RRE on unoccluded pixels)."""
    return (relative_reconstruction_error(V_clean, W, H),
            masked_relative_reconstruction_error(V_clean, W, H, mask),
            masked_relative_reconstruction_error(V_clean, W, H, ~mask))


def trace(V, V_clean, mask, rank, seed, lam, total, step):
    """Yield (iteration, objective, rre, rre_occ, rre_unocc, elapsed) every step."""
    W, H = _shared_initialization(V, rank, seed)
    objective, start = [], time.perf_counter()
    for done in range(step, total + 1, step):
        if lam is None:
            W, H, obj, _ = nmf_baseline(V, rank=rank, max_iter=step, tol=0.0,
                                        random_state=seed, W_init=W, H_init=H)
        else:
            W, H, _, history = fit_rnmf_l1(V, rank, lam, max_iter=step, tol=0.0,
                                           seed=seed, W_init=W, H_init=H)
            obj = history["objective"]
        objective.extend(obj)
        yield (done, objective[-1], *_rres(V_clean, W, H, mask),
               time.perf_counter() - start, objective)


def run(args):
    """Fit every method without early stopping and save the trajectories."""
    config = DATASETS[args.dataset]
    V_all, labels, image_shape = load_data(args.data_root / config["folder"],
                                           reduce=config["reduce"])
    rank = args.rank or int(np.unique(labels).size)
    lambdas = [None] + [float(v) for v in args.lambdas.split(",")]
    b, m = (int(v) for v in args.setting.split(":"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = Path(f"{_base(args, b, m)}_trajectory.csv")
    stops_path = Path(f"{_base(args, b, m)}_rule_stops.csv")
    print(f"{args.dataset}: rank={rank}, b={b}, m={m}, {args.repeats} repeats", flush=True)
    rows, stops, start = [], [], time.perf_counter()
    for repeat in range(args.repeats):
        V_clean, _, _ = sample_columns(V_all, labels, fraction=0.9, seed=SAMPLE_SEED + repeat)
        V, mask, _ = apply_occlusion(V_clean, image_shape, b, m, seed=CORRUPTION_SEED + repeat)
        for lam in lambdas:
            method = "Frobenius-NMF" if lam is None else "RNMF-L1"
            for it, obj, rre, occ, unocc, elapsed, objective in trace(
                    V, V_clean, mask, rank, INIT_SEED + repeat, lam, args.iterations, args.step):
                rows.append({"dataset": args.dataset, "block_size": b, "num_blocks": m,
                             "repeat": repeat, "method": method,
                             "lambda": "" if lam is None else lam, "iteration": it,
                             "objective": obj, "rre": rre, "rre_occluded": occ,
                             "rre_unoccluded": unocc, "elapsed_seconds": elapsed})
            stop = window_stop(objective, config["tol"])
            stops.append({"repeat": repeat, "method": method,
                          "lambda": "" if lam is None else lam, "rule_stop_iteration": stop})
            for p, data in ((path, rows), (stops_path, stops)):
                with p.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
            label = method if lam is None else f"{method} lam={lam:g}"
            best = min((r for r in rows if r["repeat"] == repeat and r["method"] == method
                        and r["lambda"] == ("" if lam is None else lam)), key=lambda r: r["rre"])
            print(f"r={repeat} {label:22} RRE@{args.iterations}={rows[-1]['rre']:.4f} "
                  f"min RRE={best['rre']:.4f}@{best['iteration']} rule stop@{stop} "
                  f"{elapsed:5.1f}s total={(time.perf_counter() - start) / 60:4.1f}min", flush=True)


def plot(args):
    """Plot mean ± std (ddof=1) RRE trajectories from the saved CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, m = (int(v) for v in args.setting.split(":"))
    base = _base(args, b, m)
    with Path(f"{base}_trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with Path(f"{base}_rule_stops.csv").open(newline="") as stream:
        stops = list(csv.DictReader(stream))
    keys = []
    for r in rows:
        key = (r["method"], r["lambda"])
        if key not in keys:
            keys.append(key)
    palette = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for (method, lam), color in zip(keys, palette):
        sel = [r for r in rows if r["method"] == method and r["lambda"] == lam]
        its = sorted({int(r["iteration"]) for r in sel})
        label = "Frobenius NMF" if method == "Frobenius-NMF" else f"RNMF-L1 λ={float(lam):g}"
        style = "--" if method == "Frobenius-NMF" else "-"
        for ax, key in zip(axes, ("rre", "rre_occluded")):
            vals = np.array([[float(r[key]) for r in sel if int(r["iteration"]) == it] for it in its])
            mu = vals.mean(axis=1)
            sd = vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros_like(mu)
            ax.plot(its, mu, style, color=color, linewidth=1.8, label=label)
            ax.fill_between(its, mu - sd, mu + sd, color=color, alpha=0.12, linewidth=0)
            stop_its = [s["rule_stop_iteration"] for s in stops
                        if s["method"] == method and s["lambda"] == lam and s["rule_stop_iteration"]]
            if stop_its:
                it_stop = int(np.median([int(s) for s in stop_its]))
                y = np.interp(it_stop, its, mu)
                ax.plot([it_stop], [y], marker="v", color=color, markersize=7,
                        markeredgecolor="white", linestyle="none")
    for ax, title in zip(axes, ("RRE, all pixels", "RRE, occluded pixels only")):
        ax.set_xlabel("Iteration")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("RRE against clean data")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    n = len({r["repeat"] for r in rows})
    fig.text(0.01, -0.04, f"{args.dataset.upper()}, b={b}, m={m}, rank k={args.rank or 'n_classes'}, mean ± std over {n} repeats "
             "(tuning seeds). No early stopping; ▼ marks where the current window-5 / "
             "min_iter-200 rule stops (median over repeats).", fontsize=8, color="#52514e")
    fig.tight_layout()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure = args.figure_dir / f"{_name(args, b, m)}_rre_trajectory"
    for ext in ("png", "pdf"):
        fig.savefig(f"{figure}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {figure}.png")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--setting", default="8:2", help="b:m")
    parser.add_argument("--lambdas", default="0.005,0.01,0.02,0.03,0.05,0.1,0.3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR,
                        help="folder containing ORL/ and CroppedYaleB/")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="folder for the trajectory CSVs")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--rank", type=int, default=None,
                        help="factorisation rank k (default: number of classes, as in main.py)")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        run(args)
    plot(args)


if __name__ == "__main__":
    main()
