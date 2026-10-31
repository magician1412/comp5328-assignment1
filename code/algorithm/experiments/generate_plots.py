"""Plots and a LaTeX table for one main.py run folder (ORL or YaleB).

Reads <run>/summary.csv and <run>/objective_trace.csv, written by main.py,
and writes into <run>/plots/:
  * <dataset>_rre_sweeps.{png,pdf}: RRE vs b (m = 1) and RRE vs m (b = 8),
    mean ± std over repeats, both methods;
  * <dataset>_objective.{png,pdf}: objective vs iteration for repeat 0;
  * <dataset>_summary_table.tex: RRE, paired RRE difference, ACC and NMI.
No face images are drawn.

Run from any directory (default: the newest folder in code/results):
    python code/algorithm/experiments/generate_plots.py
    python code/algorithm/experiments/generate_plots.py --run-dir code/results/<run>
"""

import argparse
from pathlib import Path
import sys

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

from experiments.result_plots import (  # noqa: E402
    plot_objective_traces,
    plot_rre_sweeps,
    read_rows,
    write_latex_table,
)
from utils.paths import RESULTS_DIR  # noqa: E402


def latest_run_dir(results_dir=RESULTS_DIR):
    """Return the newest run folder (by name, i.e. by start time) in results_dir."""
    runs = sorted(p for p in Path(results_dir).iterdir()
                  if p.is_dir() and (p / "summary.csv").exists())
    if not runs:
        raise FileNotFoundError(f"no run folder with summary.csv in {results_dir}")
    return runs[-1]


def generate(run_dir):
    """Write all plots and the table for one run folder; return the paths."""
    import matplotlib

    matplotlib.use("Agg")

    run_dir = Path(run_dir)
    summary = read_rows(run_dir / "summary.csv")
    order = {"Frobenius-NMF": 0, "RNMF-L1": 1}
    summary.sort(key=lambda r: (int(r["num_blocks"]) > 0, int(r["num_blocks"]),
                                int(r["block_size"]), order.get(r["algorithm"], 2)))
    dataset = summary[0]["dataset"]
    stem = run_dir / "plots" / dataset.lower()
    paths = plot_rre_sweeps(summary, f"{stem}_rre_sweeps")
    trace_path = run_dir / "objective_trace.csv"
    if trace_path.exists():
        paths += plot_objective_traces(read_rows(trace_path), f"{stem}_objective",
                                       dataset=dataset)
    paths.append(write_latex_table(summary, f"{stem}_summary_table.tex"))
    return paths


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="run folder written by main.py (default: newest in "
                             "code/results)")
    args = parser.parse_args(argv)
    for path in generate(args.run_dir or latest_run_dir()):
        print(path)


if __name__ == "__main__":
    main()
