"""Fair, reproducible experiment runner and CSV aggregation."""

from dataclasses import dataclass, field
from pathlib import Path
import csv
import time

import numpy as np

from utils.evaluation_metrics import (
    clustering_scores,
    masked_relative_reconstruction_error,
    relative_reconstruction_error,
)


@dataclass(frozen=True)
class FitResult:
    """Normalized output returned by an algorithm adapter."""

    W: np.ndarray
    H: np.ndarray
    history: dict = field(default_factory=dict)
    S: np.ndarray | None = None


@dataclass(frozen=True)
class AlgorithmSpec:
    """Name, fitting callback, and report metadata for one algorithm."""

    name: str
    fit: object
    parameters: dict = field(default_factory=dict)

def _shared_initialization(V, rank, seed):
    """Return the initial (W, H) shared by both methods for one seed.

    Entries are uniform on [0, 1) scaled by sqrt(mean(V) / rank), so that
    W @ H starts at the scale of V.
    """
    rng = np.random.default_rng(seed)

    scale = np.sqrt(V.mean() / rank)

    W_init = rng.random((V.shape[0], rank)) * scale
    H_init = rng.random((rank, V.shape[1])) * scale

    return W_init, H_init

RAW_FIELDS = [
    "dataset", "reduce", "algorithm", "rank", "lambda", "max_iter", "tol",
    "block_size", "num_blocks", "repeat", "sample_fraction", "sample_seed",
    "corruption_seed", "init_seed", "n_features", "n_samples",
    "corrupted_fraction", "rre", "rre_occluded", "rre_unoccluded", "accuracy", "nmi", "runtime_seconds",
    "iterations", "converged", "objective_initial", "objective_final",
]


def baseline_spec(max_iter=500, tol=1e-5):
    """Return an adapter for the standard ``nmf_baseline`` implementation."""
    from nmf.nmf_baseline import nmf_baseline

    def fit(V, rank, seed):
        """Fit Frobenius NMF from the shared initialisation for ``seed``."""
        W_init, H_init = _shared_initialization(V, rank, seed)

        W, H, objective, converged = nmf_baseline(
            V,
            rank=rank,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            W_init=W_init,
            H_init=H_init,
        )
        return FitResult(
            W=W, H=H,
            history={
                "objective": list(objective), "n_iter": len(objective),
                "converged": converged,
            },
        )

    return AlgorithmSpec(
        "Frobenius-NMF", fit,
        {"max_iter": max_iter, "tol": tol, "lambda": None},
    )


def rnmf_l1_spec(lam, max_iter=500, tol=1e-5):
    """Return an adapter for the ``fit_rnmf_l1`` implementation.

    The initial factors reproduce ``nmf_baseline`` initialization exactly so
    paired runs compare the objectives from the same starting point.
    """
    from nmf.nmf_rnmf_l1 import fit_rnmf_l1

    def fit(V, rank, seed):
        """Fit RNMF-L1 from the shared initialisation for ``seed``."""
        W_init, H_init = _shared_initialization(V, rank, seed)

        W, H, S, history = fit_rnmf_l1(
            V,
            rank=rank,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            W_init=W_init,
            H_init=H_init,
        )
        return FitResult(W=W, H=H, S=S, history=history)

    return AlgorithmSpec(
        "RNMF-L1", fit,
        {"max_iter": max_iter, "tol": tol, "lambda": float(lam)},
    )


def _history_value(history, key, default):
    """Return ``history[key]`` as a plain Python value (``default`` if absent)."""
    value = history.get(key, default)
    return value.item() if isinstance(value, np.generic) else value


def run_condition(
    V_clean, labels, image_shape, algorithms, *, dataset, reduce, rank,
    block_size, num_blocks, repeat, sample_fraction=0.9, sample_seed=100,
    corruption_seed=200, init_seed=300, clustering=True, kmeans_n_init=10,
    sampler=None, occluder=None, return_artifacts=False, verbose=False,
):
    """Run all algorithms on one identical sampled and corrupted condition.

    Seeds are offset by ``repeat``.  Sampling and corruption are performed once
    before the algorithm loop, guaranteeing paired comparisons.  Each returned
    row holds the overall RRE and the RRE restricted to occluded and to
    unoccluded pixels (``rre_occluded`` is NaN when nothing is occluded).
    KMeans uses ``random_state = init_seed + repeat`` and ``kmeans_n_init``.
    With ``return_artifacts`` the clean and occluded matrices, each method's
    reconstruction W @ H and its objective history are returned as well.
    """
    current_sample_seed = int(sample_seed) + int(repeat)
    current_corruption_seed = int(corruption_seed) + int(repeat)
    current_init_seed = int(init_seed) + int(repeat)
    if sampler is None:
        from utils.sampling import sample_columns
        sampler = sample_columns
    if occluder is None:
        from utils.occlusion import apply_occlusion
        occluder = apply_occlusion
    V_sub, labels_sub, _ = sampler(
        V_clean, labels, fraction=sample_fraction, seed=current_sample_seed,
    )
    V_noisy, mask, _ = occluder(
        V_sub, image_shape, block_size=block_size,
        num_blocks=num_blocks, seed=current_corruption_seed,
    )
    corrupted_fraction = float(mask.mean())
    rows = []

    artifacts = None
    if return_artifacts:
        artifacts = {
            "clean": V_sub.copy(),
            "occluded": V_noisy.copy(),
            "reconstructions": {},
            "objectives": {},
        }

    for spec in algorithms:
        if verbose:
            print(
                f"    Running {spec.name}...",
                flush=True,
            )
        started = time.perf_counter()
        result = spec.fit(
            V_noisy.copy(),
            int(rank),
            current_init_seed,
        )
        measured_runtime = time.perf_counter() - started
        if not isinstance(result, FitResult):
            raise TypeError(
                f"{spec.name} adapter must return FitResult"
            )
        objective = list(
            result.history.get("objective", [])
        )
        n_iter = int(
            _history_value(
                result.history,
                "n_iter",
                len(objective),
            )
        )
        converged = bool(
            _history_value(
                result.history,
                "converged",
                False,
            )
        )
        if verbose:
            print(
                f"    Finished {spec.name} | "
                f"{measured_runtime:.1f}s | "
                f"iterations={n_iter} | "
                f"converged={converged}",
                flush=True,
            )

        if return_artifacts:
            artifacts["reconstructions"][spec.name] = result.W @ result.H
            artifacts["objectives"][spec.name] = objective

        scores = {"accuracy": np.nan, "nmi": np.nan}
        if clustering:
            scores = clustering_scores(
                result.H, labels_sub, seed=current_init_seed,
                n_init=kmeans_n_init, n_clusters=rank,
            )
        row = {
            "dataset": str(dataset), "reduce": int(reduce),
            "algorithm": spec.name, "rank": int(rank),
            "lambda": spec.parameters.get("lambda"),
            "max_iter": spec.parameters.get("max_iter"),
            "tol": spec.parameters.get("tol"),
            "block_size": int(block_size), "num_blocks": int(num_blocks),
            "repeat": int(repeat), "sample_fraction": float(sample_fraction),
            "sample_seed": current_sample_seed,
            "corruption_seed": current_corruption_seed,
            "init_seed": current_init_seed, "n_features": V_sub.shape[0],
            "n_samples": V_sub.shape[1],
            "corrupted_fraction": corrupted_fraction,
            "rre": relative_reconstruction_error(V_sub, result.W, result.H),
            "rre_occluded": masked_relative_reconstruction_error(
                V_sub, result.W, result.H, mask),
            "rre_unoccluded": masked_relative_reconstruction_error(
                V_sub, result.W, result.H, ~mask),
            "accuracy": scores["accuracy"], "nmi": scores["nmi"],
            "runtime_seconds": float(measured_runtime),
            "iterations": n_iter,
            "converged": converged,
            "objective_initial": objective[0] if objective else np.nan,
            "objective_final": objective[-1] if objective else np.nan,
        }
        rows.append(row)

    if return_artifacts:
        return rows, artifacts

    return rows


def run_sweep(
    V_clean, labels, image_shape, algorithms, settings, *, dataset, reduce,
    repeats=5, sample_fraction=0.9, rank=None, sample_seed=100,
    corruption_seed=200, init_seed=300, clustering=True, kmeans_n_init=10,
    collect_visualizations=False, visual_sample_index=0, verbose=False,
):
    """Run all occlusion settings and repeats.

    When collect_visualizations=False, return rows.

    When collect_visualizations=True, return (rows, visual_records), where
    visual_records maps each (b, m) setting to its repeat-0 record: the
    representative sample (clean, occluded and reconstructed columns) and
    the objective history of every method ("objectives").
    """
    labels = np.asarray(labels)
    if rank is None:
        rank = np.unique(labels).size
    rows = []
    visual_records = {}

    total_conditions = len(settings) * int(repeats)
    condition_number = 0

    for block_size, num_blocks in settings:
        for repeat in range(int(repeats)):

            condition_number += 1

            if verbose:
                print()
                print(
                    f"[{condition_number}/{total_conditions}] "
                    f"block_size={block_size}, "
                    f"num_blocks={num_blocks}, "
                    f"repeat={repeat + 1}/{repeats}",
                    flush=True,
                )

            collect_this_run = (collect_visualizations and repeat == 0)

            result = run_condition(V_clean, labels, image_shape, algorithms,
                dataset=dataset, reduce=reduce, rank=rank, block_size=block_size,
                num_blocks=num_blocks, repeat=repeat, sample_fraction=sample_fraction,
                sample_seed=sample_seed, corruption_seed=corruption_seed, init_seed=init_seed,
                clustering=clustering, kmeans_n_init=kmeans_n_init, return_artifacts=collect_this_run,
                verbose=verbose,
            )

            if collect_this_run:
                condition_rows, artifacts = result
                rows.extend(condition_rows)

                sample_index = int(visual_sample_index)

                if not 0 <= sample_index < artifacts["clean"].shape[1]:
                    raise ValueError("visual_sample_index is outside the sampled dataset")

                visual_records[
                    (int(block_size), int(num_blocks))
                ] = {
                    "clean": artifacts["clean"][
                        :, [sample_index]
                    ].copy(),

                    "occluded": artifacts["occluded"][
                        :, [sample_index]
                    ].copy(),

                    "reconstructions": {
                        name: reconstruction[
                            :, [sample_index]
                        ].copy()
                        for name, reconstruction
                        in artifacts["reconstructions"].items()
                    },

                    "objectives": artifacts["objectives"],
                }

            else:
                rows.extend(result)

    if collect_visualizations:
        return rows, visual_records

    return rows


def write_csv(rows, path, fieldnames=RAW_FIELDS):
    """Write dictionaries to a stable CSV schema and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sample_std(values):
    """Return the sample standard deviation (ddof=1); NaN for fewer than 2 values."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def paired_differences(rows, metric="rre", first="Frobenius-NMF", second="RNMF-L1"):
    """Return {(dataset, b, m): array of metric(first) - metric(second)}.

    Rows are paired by repeat, i.e. on the same subsample, occlusion and
    initialisation. With the defaults a positive difference means RNMF-L1
    has the lower RRE.
    """
    values = {}
    for row in rows:
        key = (row["dataset"], int(row["block_size"]), int(row["num_blocks"]))
        values.setdefault(key, {}).setdefault(row["algorithm"], {})[
            int(row["repeat"])] = float(row[metric])
    differences = {}
    for key, by_method in values.items():
        if first in by_method and second in by_method:
            repeats = sorted(set(by_method[first]) & set(by_method[second]))
            differences[key] = np.array([
                by_method[first][r] - by_method[second][r] for r in repeats
            ])
    return differences


def aggregate_rows(rows):
    """Aggregate repeated rows into mean/std summaries by test condition.

    Standard deviations are sample standard deviations over repeats (ddof=1;
    NaN with a single repeat). Every row of a setting also carries the paired
    difference RRE(Frobenius-NMF) - RRE(RNMF-L1) over repeats
    (``paired_rre_diff_mean`` / ``_std`` / ``_n``).
    """
    group_fields = [
        "dataset", "reduce", "algorithm", "rank", "lambda", "max_iter", "tol", "block_size",
        "num_blocks", "sample_fraction", "n_features", "n_samples",
    ]
    metrics = [
        "corrupted_fraction", "rre", "rre_occluded", "rre_unoccluded", "accuracy",
        "nmi", "runtime_seconds", "iterations", "objective_final",
    ]
    groups = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)
    differences = paired_differences(rows)
    output = []
    def order(item):
        """Sort by dataset, algorithm, then numerically by (b, m)."""
        fields = dict(zip(group_fields, item[0]))
        return (str(fields["dataset"]), str(fields["algorithm"]),
                int(fields["block_size"]), int(fields["num_blocks"]))

    for key, members in sorted(groups.items(), key=order):
        summary = dict(zip(group_fields, key))
        summary["repeats"] = len(members)
        summary["converged_runs"] = sum(bool(row["converged"]) for row in members)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members], dtype=float)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else np.nan
            )
            summary[f"{metric}_std"] = sample_std(finite)
        diff = differences.get(
            (summary["dataset"], int(summary["block_size"]), int(summary["num_blocks"])),
            np.array([]),
        )
        summary["paired_rre_diff_mean"] = float(diff.mean()) if diff.size else np.nan
        summary["paired_rre_diff_std"] = sample_std(diff)
        summary["paired_rre_diff_n"] = int(diff.size)
        output.append(summary)
    return output


OBJECTIVE_TRACE_FIELDS = ["setting", "block_size", "num_blocks", "method",
                          "iteration", "objective"]


def objective_trace_rows(visual_records):
    """Return long-format rows (setting, method, iteration, objective).

    ``visual_records`` is the second value returned by ``run_sweep`` with
    ``collect_visualizations=True``; iterations are numbered from 1.
    """
    rows = []
    for (block_size, num_blocks), record in visual_records.items():
        for method, objective in record["objectives"].items():
            rows.extend({
                "setting": f"{block_size}:{num_blocks}", "block_size": block_size,
                "num_blocks": num_blocks, "method": method,
                "iteration": iteration, "objective": value,
            } for iteration, value in enumerate(objective, start=1))
    return rows
