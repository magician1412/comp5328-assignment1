"""Lambda-sensitivity study for RNMF-L1 (Member 2).

Protocol (fixed before the results were seen):
  * Rank k = number of classes (ORL 40, YaleB 38), as in main.py.
  * Frobenius NMF keeps the tolerance stopping rule of main.py
    (tol ORL 1e-3 / YaleB 5e-4, min_iter 200, max_iter 3000); it converges
    within a few hundred iterations and its RRE is then stable.
  * RNMF-L1 runs a fixed budget of N = 2000 iterations (tol = 0). The
    tolerance rule stops RNMF-L1 on an early objective plateau for small
    lambda, and its RRE depends on the iteration count, so all lambdas are
    compared at the same budget (see experiments/rre_trajectory.py).
  * lambda is chosen per dataset on ONE moderate setting (b=8, m=2); the
    no-occlusion setting (b=4, m=0) only shows the cost of small lambda.
  * Seeds are disjoint from the main experiment (main uses 100/200/300 + r):
    sample 1000 + r, occlusion 2000 + r, initialisation 3000 + r. Within a
    repeat every lambda and the NMF reference share the subset, the occlusion
    mask and the initial W, H.

Run from any directory (the repository root is shown):
    python -B code/algorithm/experiments/rnmf_lambda_sensitivity.py --dataset orl
    python -B code/algorithm/experiments/rnmf_lambda_sensitivity.py --dataset yaleb --settings 8:2
    python -B code/algorithm/experiments/rnmf_lambda_sensitivity.py --dataset yaleb \
        --settings 4:0 --repeats 3 --append

Outputs in experiments/lambda_study_results/lambda_sensitivity/:
<dataset>_raw.csv (one row per fit, rewritten after every fit),
<dataset>_examples_b<b>_m<m>.npz (four image columns from repeat 0 for the
reconstruction figure; gitignored because it holds faces) and
<dataset>_metadata.json. Summaries, figures and the lambda choice come from
plot_lambda_sensitivity.py.
"""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

import numpy as np  # noqa: E402

from experiments.experiment import baseline_spec, rnmf_l1_spec  # noqa: E402
from experiments.run_info import git_state  # noqa: E402
from utils.data_loader import load_data  # noqa: E402
from utils.evaluation_metrics import (  # noqa: E402
    masked_relative_reconstruction_error,
    relative_reconstruction_error,
)
from utils.occlusion import apply_occlusion  # noqa: E402
from utils.paths import DATA_DIR  # noqa: E402
from utils.sampling import sample_columns  # noqa: E402

# folder / reduce / tol mirror DATASET_CONFIGS in main.py.
DATASETS = {
    "orl": {"folder": "ORL", "reduce": 3, "tol": 1e-3,
            "lambdas": "0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.1,0.3"},
    "yaleb": {"folder": "CroppedYaleB", "reduce": 4, "tol": 5e-4,
              "lambdas": "0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.1"},
}
OUTPUT_DIR = Path(__file__).resolve().parent / "lambda_study_results" / "lambda_sensitivity"
SAMPLE_SEED, CORRUPTION_SEED, INIT_SEED = 1000, 2000, 3000
SAMPLE_FRACTION = 0.9
N_EXAMPLES = 4

FIELDS = [
    "dataset", "reduce", "rank", "algorithm", "lambda", "block_size",
    "num_blocks", "repeat", "sample_seed", "corruption_seed", "init_seed",
    "max_iter", "tol", "corrupted_fraction", "rre", "rre_occluded",
    "rre_unoccluded", "s_nonzero", "s_mass_in_mask", "mask_recall",
    "iterations", "converged", "runtime_seconds",
]


def _settings(text):
    """Parse "b:m,b:m" into a list of (b, m) tuples."""
    return [tuple(int(v) for v in item.split(":")) for item in text.split(",")]


def _check_factors(result, V):
    """Fail loudly if the pipeline returns malformed factors."""
    d, n = V.shape
    k = result.W.shape[1]
    if result.W.shape != (d, k) or result.H.shape != (k, n):
        raise ValueError(f"unexpected shapes W{result.W.shape} H{result.H.shape}")
    if result.S is not None and result.S.shape != (d, n):
        raise ValueError(f"unexpected S shape {result.S.shape}")
    for name, M in (("W", result.W), ("H", result.H), ("S", result.S)):
        if M is not None and not np.isfinite(M).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")
    if (result.W < 0).any() or (result.H < 0).any():
        raise ValueError("W or H has negative entries")


def _row(common, name, lam, spec_params, result, V_clean, mask, runtime):
    """Return one raw-results row for a fitted method."""
    S = result.S
    row = dict(common, algorithm=name, **{"lambda": "" if lam is None else lam})
    row.update(
        max_iter=spec_params["max_iter"], tol=spec_params["tol"],
        rre=relative_reconstruction_error(V_clean, result.W, result.H),
        rre_occluded=masked_relative_reconstruction_error(
            V_clean, result.W, result.H, mask),
        rre_unoccluded=masked_relative_reconstruction_error(
            V_clean, result.W, result.H, ~mask),
        iterations=result.history["n_iter"],
        converged=bool(result.history["converged"]),
        runtime_seconds=runtime,
    )
    if S is None:
        row.update(s_nonzero="", s_mass_in_mask="", mask_recall="")
    else:
        nonzero = np.abs(S) > 1e-8
        mass = np.abs(S).sum()
        row.update(
            s_nonzero=float(nonzero.mean()),
            s_mass_in_mask=(float(np.abs(S)[mask].sum() / mass)
                            if mass > 0 and mask.any() else float("nan")),
            mask_recall=float((nonzero & mask).sum() / mask.sum()) if mask.any() else float("nan"),
        )
    return row


def main():
    """Run the lambda study for one dataset and write raw rows and metadata."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--lambdas", default=None, help="comma list; default per dataset")
    parser.add_argument("--settings", default="8:2,4:0", help="b:m pairs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rnmf-iterations", type=int, default=2000,
                        help="fixed RNMF-L1 budget N (tol = 0)")
    parser.add_argument("--nmf-max-iter", type=int, default=3000)
    parser.add_argument("--append", action="store_true",
                        help="keep rows already in <dataset>_raw.csv")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR,
                        help="folder containing ORL/ and CroppedYaleB/")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    lambdas = [float(v) for v in (args.lambdas or config["lambdas"]).split(",")]
    settings = _settings(args.settings)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{args.dataset}_raw.csv"
    rows = []
    if args.append and raw_path.exists():
        with raw_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))

    V_all, labels, image_shape = load_data(args.data_root / config["folder"],
                                         reduce=config["reduce"])
    rank = int(np.unique(labels).size)
    nmf_params = {"max_iter": args.nmf_max_iter, "tol": config["tol"]}
    rnmf_params = {"max_iter": args.rnmf_iterations, "tol": 0.0}
    algorithms = [("Frobenius-NMF", None, nmf_params, baseline_spec(**nmf_params))]
    algorithms += [("RNMF-L1", lam, rnmf_params, rnmf_l1_spec(lam=lam, **rnmf_params))
                   for lam in lambdas]

    total_fits = len(settings) * args.repeats * len(algorithms)
    print(f"{args.dataset}: V={V_all.shape}, image={image_shape}, rank={rank}, "
          f"settings={settings}, repeats={args.repeats}, NMF tol={config['tol']} "
          f"max_iter={args.nmf_max_iter}, RNMF fixed N={args.rnmf_iterations}, "
          f"lambdas={lambdas}: {total_fits} fits", flush=True)

    fit_no, start = 0, time.perf_counter()
    for block_size, num_blocks in settings:
        examples = {}
        for repeat in range(args.repeats):
            seeds = (SAMPLE_SEED + repeat, CORRUPTION_SEED + repeat, INIT_SEED + repeat)
            V_clean, _, _ = sample_columns(V_all, labels, fraction=SAMPLE_FRACTION, seed=seeds[0])
            V, mask, _ = apply_occlusion(V_clean, image_shape, block_size, num_blocks, seed=seeds[1])
            common = {
                "dataset": args.dataset, "reduce": config["reduce"], "rank": rank,
                "block_size": block_size, "num_blocks": num_blocks, "repeat": repeat,
                "sample_seed": seeds[0], "corruption_seed": seeds[1], "init_seed": seeds[2],
                "corrupted_fraction": float(mask.mean()),
            }
            cols = np.arange(N_EXAMPLES) * (V.shape[1] // N_EXAMPLES)
            if repeat == 0:
                examples.update(clean=V_clean[:, cols], noisy=V[:, cols], mask=mask[:, cols])
            for name, lam, params, spec in algorithms:
                fit_no += 1
                t_fit = time.perf_counter()
                result = spec.fit(V, rank, seeds[2])
                runtime = time.perf_counter() - t_fit
                _check_factors(result, V)
                row = _row(common, name, lam, params, result, V_clean, mask, runtime)
                rows.append(row)
                with raw_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                if repeat == 0:
                    key = "nmf" if lam is None else f"lam_{lam:g}"
                    examples[f"WH_{key}"] = (result.W @ result.H)[:, cols]
                    if result.S is not None:
                        examples[f"S_{key}"] = result.S[:, cols]
                label = name if lam is None else f"{name} lam={lam:g}"
                print(f"[{fit_no}/{total_fits}] b={block_size} m={num_blocks} r={repeat} "
                      f"{label:22} RRE={row['rre']:.4f} occ={row['rre_occluded']:.4f} "
                      f"it={row['iterations']:4d} {runtime:6.1f}s "
                      f"elapsed={(time.perf_counter() - start) / 60:5.1f}min", flush=True)
        np.savez_compressed(out / f"{args.dataset}_examples_b{block_size}_m{num_blocks}.npz",
                            image_shape=np.array(image_shape), columns=cols, **examples)

    total = time.perf_counter() - start
    commit = git_state()["commit_short"]
    meta_path = out / f"{args.dataset}_metadata.json"
    runs = json.loads(meta_path.read_text())["runs"] if args.append and meta_path.exists() else []
    runs.append({
        "git_commit": commit, "settings": settings, "repeats": args.repeats,
        "lambdas": lambdas, "n_fits": total_fits,
        "wall_minutes": round(total / 60, 2),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    metadata = {
        "dataset": args.dataset, "rank": rank, "reduce": config["reduce"],
        "tuning_setting": [8, 2],
        "seeds": {"sample": SAMPLE_SEED, "corruption": CORRUPTION_SEED,
                  "init": INIT_SEED, "rule": "base + repeat index"},
        "frobenius_nmf": {"tol": config["tol"], "max_iter": args.nmf_max_iter,
                          "rule": "window-5 cumulative relative change < tol, min_iter 200"},
        "rnmf_l1": {"fixed_iterations": args.rnmf_iterations, "tol": 0.0},
        "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.platform(), "runs": runs,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"done: {total_fits} fits in {total / 60:.1f} min -> {raw_path}", flush=True)


if __name__ == "__main__":
    main()
