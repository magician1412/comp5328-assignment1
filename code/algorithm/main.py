"""Main experiment: Frobenius NMF vs RNMF-L1 under block occlusion.

For one dataset, every occlusion setting (b, m) in SETTINGS and every repeat
r draws one 90% column subsample (seed SAMPLE_SEED + r) and one occlusion
(seed CORRUPTION_SEED + r); both methods are fitted to that same corrupted
matrix from the same initial W, H (seed INIT_SEED + r) at rank = number of
classes, and are evaluated against the matching clean columns.

Usage (from any directory):
    python code/algorithm/main.py --dataset orl
    python code/algorithm/main.py --dataset yaleb
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.experiment import (
    OBJECTIVE_TRACE_FIELDS,
    aggregate_rows,
    baseline_spec,
    objective_trace_rows,
    rnmf_l1_spec,
    run_sweep,
    write_csv,
)
from experiments.generate_plots import generate as generate_plots
from experiments.run_info import (
    command_line,
    environment_versions,
    git_state,
    make_run_dir,
    timestamp,
)
from utils.data_loader import load_data
from utils.paths import DATA_DIR, RESULTS_DIR
from utils.visualize import plot_reconstruction_comparison

# ==========================================================
# Agreed experimental protocol (fixed before the final sweep)
# ==========================================================

# Occlusion settings (b, m): b = block edge length, m = number of blocks.
SETTINGS = [
    (4, 0),  # clean reference (no occlusion)
    (4, 1), (8, 1), (12, 1), (16, 1),  # m = 1 fixed, block size b varies
    (8, 2), (8, 3),  # b = 8 fixed, block count m varies (with (8, 1))
]
REPEATS = 5
SAMPLE_FRACTION = 0.9
# Seeds for repeat r are base + r.
SAMPLE_SEED = 100
CORRUPTION_SEED = 200
INIT_SEED = 300
# Column of the repeat-0 subsample shown in the reconstruction figures.
VISUAL_SAMPLE_INDEX = 0
# KMeans on H.T (evaluation only): random_state = INIT_SEED + r.
KMEANS_N_INIT = 10

# Frobenius NMF stops when the cumulative relative objective change over the
# last 5 iterations is below tol, but never before NMF_MIN_ITER iterations.
# NMF_MIN_ITER documents the value built into nmf_baseline (it is not a
# solver argument); tests/test_protocol.py checks that they agree.
NMF_MIN_ITER = 200
# RNMF-L1 runs a fixed budget: tol = 0 disables the stopping rule, because
# the rule stops it on an early objective plateau for small lambda and
# lambda was selected at this budget (experiments/rnmf_lambda_sensitivity.py).
DATASET_CONFIGS = {
    "orl": {
        "folder": "ORL",
        "name": "ORL",
        "reduce": 3,
        "nmf": {"tol": 1e-3, "max_iter": 2000, "min_iter": NMF_MIN_ITER},
        "rnmf": {"lam": 0.01, "tol": 0.0, "max_iter": 2000},
    },
    "yaleb": {
        "folder": "CroppedYaleB",
        "name": "CroppedYaleB",
        "reduce": 4,
        "nmf": {"tol": 5e-4, "max_iter": 2000, "min_iter": NMF_MIN_ITER},
        "rnmf": {"lam": 0.03, "tol": 0.0, "max_iter": 2000},
    },
}


def parse_args(argv=None):
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Run NMF robustness experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="orl",
                        help="Dataset to run: orl or yaleb.")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"Number of repeated experimental runs (default {REPEATS}).")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR,
                        help="Folder containing ORL/ and CroppedYaleB/ (default: code/data).")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                        help="Parent folder of the run folders (default: code/results).")
    return parser.parse_args(argv)


def build_metadata(args, config, data_info, git, started_at):
    """Return the metadata dictionary describing one run (times added later)."""
    nmf_cfg, rnmf_cfg = config["nmf"], config["rnmf"]
    return {
        "dataset": args.dataset,
        "dataset_name": config["name"],
        "data_dir": str(args.data_root / config["folder"]),
        "reduce": config["reduce"],
        **data_info,
        "repeats": args.repeats,
        "sample_fraction": SAMPLE_FRACTION,
        "settings": [list(setting) for setting in SETTINGS],
        "seeds": {"sample": SAMPLE_SEED, "corruption": CORRUPTION_SEED,
                  "init": INIT_SEED, "rule": "base + repeat index r"},
        "methods": {
            "Frobenius-NMF": {
                "tol": nmf_cfg["tol"], "max_iter": nmf_cfg["max_iter"],
                "min_iter": nmf_cfg["min_iter"],
                "stopping": "cumulative relative objective change over the "
                            "last 5 iterations < tol, after min_iter",
            },
            "RNMF-L1": {
                "lambda": rnmf_cfg["lam"], "tol": rnmf_cfg["tol"],
                "max_iter": rnmf_cfg["max_iter"], "min_iter": NMF_MIN_ITER,
                "stopping": "fixed budget: tol = 0 never triggers the rule",
            },
        },
        "initialisation": "shared W, H for both methods: uniform [0, 1) x "
                          "sqrt(mean(V) / rank), default_rng(init seed)",
        "kmeans": {"n_init": KMEANS_N_INIT, "random_state": "init seed (INIT_SEED + r)"},
        "statistics": {
            "std": "sample standard deviation over repeats (ddof=1); NaN with one repeat",
            "paired_rre_diff": "per repeat RRE(Frobenius-NMF) - RRE(RNMF-L1) on the "
                               "same subsample, occlusion and initialisation; "
                               "positive means RNMF-L1 is better",
            "rre_occluded": "||(V_clean - WH)[mask]||_F / ||V_clean[mask]||_F; "
                            "NaN when m = 0",
            "rre_unoccluded": "the same on ~mask",
            "objective_trace": "objective after every iteration, repeat 0 only",
        },
        "git": git,
        "versions": environment_versions(),
        "command": command_line(),
        "started": timestamp(started_at),
    }


def save_reconstruction_figures(visual_records, image_shape, dataset_name, out_dir):
    """Save the clean / occluded / NMF / RNMF-L1 figure for every setting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (block_size, num_blocks), visual in visual_records.items():
        path = out_dir / f"{dataset_name.lower()}_reconstruction_b{block_size}_m{num_blocks}.png"
        fig, _ = plot_reconstruction_comparison(
            visual["clean"], visual["occluded"],
            visual["reconstructions"]["Frobenius-NMF"],
            visual["reconstructions"]["RNMF-L1"],
            image_shape,
            sample_index=0,  # each record stores exactly one selected column
            title=f"{dataset_name} Reconstruction Comparison: b={block_size}, m={num_blocks}",
            save_path=path, baseline_name="Baseline NMF", rnmf_name="RNMF-L1",
        )
        plt.close(fig)


def print_summary(summary):
    """Print mean +/- std RRE, iterations and runtime per setting and method."""
    print(f"{'method':<14}{'b':>4}{'m':>4}  {'RRE mean +/- std':<20}"
          f"{'NMF-RNMF diff':<20}{'iterations':>10}{'seconds':>9}")
    for row in summary:
        print(f"{row['algorithm']:<14}{row['block_size']:>4}{row['num_blocks']:>4}  "
              f"{row['rre_mean']:.4f} +/- {row['rre_std']:<7.4f}  "
              f"{row['paired_rre_diff_mean']:+.4f} +/- {row['paired_rre_diff_std']:<6.4f}"
              f"{row['iterations_mean']:>10.0f}{row['runtime_seconds_mean']:>9.1f}")


def main(argv=None):
    """Run the sweep for one dataset and write a new run folder."""
    args = parse_args(argv)
    config = DATASET_CONFIGS[args.dataset]
    dataset_name = config["name"]
    started_at, start = datetime.now(), time.perf_counter()
    git = git_state()
    run_dir = make_run_dir(args.out_dir, started_at, git["commit_short"], args.dataset)
    print(f"Run folder: {run_dir}")

    V_clean, labels, image_shape = load_data(args.data_root / config["folder"],
                                             reduce=config["reduce"])
    rank = len(set(labels.tolist()))  # rank = number of classes
    print(f"{dataset_name}: V_clean {V_clean.shape}, image {image_shape}, "
          f"{rank} classes")

    metadata = build_metadata(args, config, {
        "image_shape": list(image_shape), "n_features": V_clean.shape[0],
        "n_images": V_clean.shape[1], "n_classes": rank, "rank": rank,
    }, git, started_at)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    nmf_cfg, rnmf_cfg = config["nmf"], config["rnmf"]
    algorithms = [
        baseline_spec(max_iter=nmf_cfg["max_iter"], tol=nmf_cfg["tol"]),
        rnmf_l1_spec(lam=rnmf_cfg["lam"], max_iter=rnmf_cfg["max_iter"],
                     tol=rnmf_cfg["tol"]),
    ]
    rows, visual_records = run_sweep(
        V_clean, labels, image_shape, algorithms, SETTINGS,
        dataset=dataset_name, reduce=config["reduce"], repeats=args.repeats,
        sample_fraction=SAMPLE_FRACTION, rank=rank, clustering=True,
        sample_seed=SAMPLE_SEED, corruption_seed=CORRUPTION_SEED,
        init_seed=INIT_SEED, kmeans_n_init=KMEANS_N_INIT,
        collect_visualizations=True, visual_sample_index=VISUAL_SAMPLE_INDEX,
        verbose=True,
    )

    summary = aggregate_rows(rows)
    raw_path = write_csv(rows, run_dir / "raw_results.csv")
    summary_path = write_csv(summary, run_dir / "summary.csv",
                             fieldnames=list(summary[0]))
    write_csv(objective_trace_rows(visual_records), run_dir / "objective_trace.csv",
              fieldnames=OBJECTIVE_TRACE_FIELDS)
    save_reconstruction_figures(visual_records, image_shape, dataset_name,
                                run_dir / "reconstructions")
    generate_plots(run_dir)

    finished_at = datetime.now()
    metadata.update(finished=timestamp(finished_at),
                    runtime_seconds=round(time.perf_counter() - start, 2),
                    outputs=sorted(str(path.relative_to(run_dir))
                                   for path in run_dir.rglob("*") if path.is_file()))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print_summary(summary)
    print(f"\nRaw results: {raw_path}\nSummary: {summary_path}\n"
          f"Total runtime: {metadata['runtime_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
