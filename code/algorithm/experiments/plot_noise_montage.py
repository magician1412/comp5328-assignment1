"""Report figures of the occlusion noise used by the main experiment.

For each dataset this writes, as PDF (for LaTeX) and PNG:
  * <dataset>_noise_montage: one row with the seven main.py settings
    b:m = 4:0, 4:1, 8:1, 12:1, 16:1, 8:2, 8:3, showing the SAME face in
    every panel (4:0 is titled "Clean (control)");
  * <dataset>_occlusion_pair_b8_m2: clean / occluded / mask at b=8, m=2,
    drawn by utils.visualize.plot_occlusion_pair.

Faithfulness: the noise is regenerated with the experiment's own functions
and arguments, not reimplemented. For repeat r, experiment.run_condition calls
    sample_columns(V, labels, fraction=sample_fraction, seed=sample_seed + r)
    apply_occlusion(V_sub, image_shape, block_size=b, num_blocks=m,
                    seed=corruption_seed + r)
with main.py's SAMPLE_FRACTION = 0.9, SAMPLE_SEED = 100 and
CORRUPTION_SEED = 200. This script uses repeat r = 0 (seeds 100 and 200) and
shows column 0 of the sampled matrix, the same image main.py uses for its
reconstruction figures (VISUAL_SAMPLE_INDEX = 0). The loader arguments
(folder, reduce) match main.DATASET_CONFIGS, and the image shape comes from
the loader. The values are duplicated below so that this script does not
import main.py; tests/test_noise_montage.py checks that they still match.

With --coverage the script writes no figures. Instead, for every setting and
repeats r = 0..COVERAGE_REPEATS-1 (main.REPEATS, the --repeats default), it computes
the mean occlusion coverage mask.mean() from the same experiment_noise call,
prints mean +/- sample std across repeats in percent, and saves
<out-dir>/coverage.csv.

Run from anywhere (defaults are resolved from this file's location):
    python -B code/algorithm/experiments/plot_noise_montage.py --dataset all
    python -B code/algorithm/experiments/plot_noise_montage.py --dataset all --coverage
Generated images contain dataset faces: do not commit them.
"""

import argparse
import csv
from pathlib import Path
import sys

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

import numpy as np

from utils.data_loader import load_data
from utils.occlusion import apply_occlusion
from utils.paths import DATA_DIR, FIGURES_DIR
from utils.sampling import sample_columns
from utils.visualize import plot_occlusion_montage, plot_occlusion_pair

# Duplicated from main.py (SETTINGS, SAMPLE_FRACTION, SAMPLE_SEED,
# CORRUPTION_SEED, VISUAL_SAMPLE_INDEX); tests/test_noise_montage.py asserts
# that they match.
SETTINGS = [(4, 0), (4, 1), (8, 1), (12, 1), (16, 1), (8, 2), (8, 3)]
SAMPLE_FRACTION = 0.9
SAMPLE_SEED = 100
CORRUPTION_SEED = 200
REPEAT = 0
SAMPLE_INDEX = 0
PAIR_SETTING = (8, 2)
# main.REPEATS (the --repeats default); the coverage table uses r = 0..4.
COVERAGE_REPEATS = 5
COVERAGE_FIELDS = ["dataset", "block_size", "num_blocks", "repeats",
                   "mean_coverage_pct", "std_coverage_pct"]

# Dataset folder and resize factor from main.DATASET_CONFIGS.
DATASETS = {
    "orl": {"folder": "ORL", "reduce": 3},
    "yaleb": {"folder": "CroppedYaleB", "reduce": 4},
}


def experiment_noise(V, labels, image_shape, block_size, num_blocks, repeat=REPEAT):
    """Return (V_sub, V_noisy, mask) exactly as run_condition builds them for one repeat."""
    V_sub, _, _ = sample_columns(
        V, labels, fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED + repeat,
    )
    V_noisy, mask, _ = apply_occlusion(
        V_sub, image_shape, block_size=block_size, num_blocks=num_blocks,
        seed=CORRUPTION_SEED + repeat,
    )
    return V_sub, V_noisy, mask


def setting_title(block_size, num_blocks):
    """Return the panel title for one (b, m) setting."""
    return "Clean (control)" if num_blocks == 0 else f"b={block_size}, m={num_blocks}"


def make_montage(V, labels, image_shape, settings=SETTINGS,
                 sample_index=SAMPLE_INDEX, repeat=REPEAT, save_path=None):
    """Return (fig, axes): one panel per setting, all showing the same sampled image."""
    columns = []
    for block_size, num_blocks in settings:
        _, V_noisy, _ = experiment_noise(V, labels, image_shape, block_size,
                                         num_blocks, repeat)
        columns.append(V_noisy[:, sample_index])
    titles = [setting_title(b, m) for b, m in settings]
    return plot_occlusion_montage(np.column_stack(columns), image_shape, titles,
                                  save_path=save_path)


def make_pair(V, labels, image_shape, setting=PAIR_SETTING,
              sample_index=SAMPLE_INDEX, repeat=REPEAT, save_path=None):
    """Return (fig, axes) of clean / occluded / mask for one setting."""
    V_sub, V_noisy, mask = experiment_noise(V, labels, image_shape, *setting, repeat)
    return plot_occlusion_pair(V_sub, V_noisy, image_shape, sample_index,
                               mask=mask, save_path=save_path)


def coverage_by_setting(V, labels, image_shape, settings=SETTINGS,
                        repeats=COVERAGE_REPEATS):
    """Return {(b, m): array of mask.mean() for repeats 0..repeats-1}."""
    coverage = {}
    for block_size, num_blocks in settings:
        coverage[(block_size, num_blocks)] = np.array([
            experiment_noise(V, labels, image_shape, block_size, num_blocks,
                             repeat)[2].mean()
            for repeat in range(repeats)
        ])
    return coverage


def coverage_rows(dataset, coverage):
    """Return one CSV row per setting: mean and sample std (ddof=1) in %, 2 d.p."""
    rows = []
    for (block_size, num_blocks), values in coverage.items():
        percent = 100.0 * values
        std = percent.std(ddof=1) if len(percent) > 1 else 0.0
        rows.append({
            "dataset": dataset, "block_size": block_size, "num_blocks": num_blocks,
            "repeats": len(percent),
            "mean_coverage_pct": f"{percent.mean():.2f}",
            "std_coverage_pct": f"{std:.2f}",
        })
    return rows


def write_coverage(rows, path):
    """Print the coverage table and save it as CSV."""
    print(f"{'dataset':<8}{'b':>4}{'m':>4}  coverage % (mean +/- std, "
          f"{rows[0]['repeats']} repeats)")
    for row in rows:
        print(f"{row['dataset']:<8}{row['block_size']:>4}{row['num_blocks']:>4}  "
              f"{row['mean_coverage_pct']:>6} +/- {row['std_coverage_pct']}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def parse_args(argv=None):
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["orl", "yaleb", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR,
                        help="folder containing ORL/ and CroppedYaleB/")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR / "noise",
                        help="output folder for the PDF and PNG figures")
    parser.add_argument("--coverage", action="store_true",
                        help="write the mean coverage table (coverage.csv) "
                             "instead of the figures")
    return parser.parse_args(argv)


def main(argv=None):
    """Write the noise figures, or with --coverage the coverage table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.coverage:
        rows = []
        for name in names:
            config = DATASETS[name]
            V, labels, image_shape = load_data(args.data_root / config["folder"],
                                               reduce=config["reduce"])
            rows += coverage_rows(name, coverage_by_setting(V, labels, image_shape))
        write_coverage(rows, args.out_dir / "coverage.csv")
        return
    for name in names:
        config = DATASETS[name]
        V, labels, image_shape = load_data(args.data_root / config["folder"],
                                           reduce=config["reduce"])
        b, m = PAIR_SETTING
        figures = [
            (f"{name}_noise_montage", make_montage(V, labels, image_shape)[0]),
            (f"{name}_occlusion_pair_b{b}_m{m}", make_pair(V, labels, image_shape)[0]),
        ]
        for stem, fig in figures:
            for suffix in (".pdf", ".png"):
                path = args.out_dir / f"{stem}{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(path)
            plt.close(fig)
        print(f"{name}: V={V.shape}, image_shape={image_shape}, repeat={REPEAT}, "
              f"sample seed={SAMPLE_SEED + REPEAT}, "
              f"corruption seed={CORRUPTION_SEED + REPEAT}, column={SAMPLE_INDEX}")


if __name__ == "__main__":
    main()
