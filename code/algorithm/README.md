# COMP5328 Assignment 1 — NMF robustness under block occlusion

This code compares two NMF algorithms on face images (ORL and Extended YaleB) that are corrupted by white square occlusion blocks:

- **Frobenius-NMF**: Lee–Seung multiplicative updates (`nmf/nmf_baseline.py`).
- **RNMF-L1** (Zhang et al., 2011): `min ||V − WH − S||_F² + λ‖S‖₁`, where S is a signed sparse error matrix (`nmf/nmf_rnmf_l1.py`).

Both solvers use only the Python standard library and NumPy. scikit-learn is used only in `utils/evaluation_metrics.py`, for KMeans and NMI. The main metric is RRE = ‖V̂ − WH‖_F / ‖V̂‖_F, computed against the clean images V̂. For RNMF-L1, S is not part of the reconstruction.

## Layout

```text
code/
├── algorithm/
│   ├── main.py              # main experiment (one dataset per run)
│   ├── nmf/                 # the two NMF solvers (NumPy only)
│   ├── utils/               # loading, sampling, occlusion, metrics, figures, paths
│   ├── experiments/         # experiment runner, plots, noise figures, lambda study
│   │   └── lambda_study_results/   # raw CSV/JSON results of the lambda study
│   ├── tests/               # pytest suite
│   ├── README.md
│   └── requirements.txt
├── data/                    # put the datasets here (empty in the submission)
├── results/                 # created by main.py: one folder per run
└── figures/                 # created by the figure scripts
```

Default paths are resolved from the location of the code, not from the current directory, so every command below works from any directory. The examples are run from the folder that contains `code/`.

## Data

Place the supplied datasets as follows:

```text
code/data/ORL/s1/1.pgm ...                 (40 subjects, 400 images)
code/data/CroppedYaleB/yaleB01/*.pgm ...   (38 subjects, 2414 usable images)
```

The loader skips `*Ambient.pgm`, `*.pgm.bad` and non-image files. Images are scaled to [0, 1] and downsampled by a factor of 3 (ORL, 37×30) or 4 (YaleB, 48×42). Use `--data-root` to read the data from another folder.

## Setup

Python 3.10 or newer is required. Install the dependencies with:

```bash
pip install -r code/algorithm/requirements.txt
```

`requirements.txt` lists the minimum versions the code needs.

**Tested with:** Python 3.10.16, numpy 1.26.4, scipy 1.11.4, matplotlib 3.7.0, pillow 11.0.0 and scikit-learn 1.5.2. With exactly these versions the results reproduce the reported numbers exactly. Newer versions also work, but numerical libraries can change floating-point details, so results may differ in the last decimals. For example, with Python 3.14.8, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pillow 12.3.0 and scikit-learn 1.9.1, a one-repeat ORL run differed from the reported RRE by at most 1.3e-13.

## Experimental protocol (fixed in `main.py`)

| | ORL | YaleB |
|---|---|---|
| Frobenius-NMF | tol = 1e-3, max_iter = 2000, min_iter = 200 | tol = 5e-4, max_iter = 2000, min_iter = 200 |
| RNMF-L1 | λ = 0.01, fixed 2000 iterations (tol = 0) | λ = 0.03, fixed 2000 iterations (tol = 0) |

- **Rank** equals the number of classes (ORL 40, YaleB 38).
- **Occlusion settings** (b:m, where b is the block edge length and m the number of blocks): 4:0 (no occlusion), 4:1, 8:1, 12:1, 16:1, 8:2 and 8:3.
- **Repeats:** 5. Each repeat uses a 90% column subsample.
- **Seeds** for repeat r: subsample 100 + r, occlusion 200 + r, initialisation 300 + r. Within a repeat, both methods are fitted to the same subsample and the same occlusion, from the same initial W and H.
- **KMeans** on Hᵀ uses `random_state = 300 + r` and `n_init = 10`.
- **Standard deviations** are sample standard deviations over repeats (ddof = 1).

Frobenius-NMF stops when the cumulative relative change of the objective over the last 5 iterations falls below tol, but never before iteration 200. RNMF-L1 always runs exactly 2000 iterations. λ was chosen by the lambda study described below.

## Commands

The commands below are run from the folder that contains `code/`.

Main experiment, one dataset per run:

```bash
python code/algorithm/main.py --dataset orl
python code/algorithm/main.py --dataset yaleb
```

Quick check with one repeat over all 7 settings:

```bash
python code/algorithm/main.py --dataset orl --repeats 1
```

Plots for a run folder. Without `--run-dir`, the newest run is used. `main.py` also produces these plots at the end of each run.

```bash
python code/algorithm/experiments/generate_plots.py --run-dir code/results/<run-folder>
```

Noise figures:

```bash
python code/algorithm/experiments/plot_noise_montage.py --dataset all
python code/algorithm/experiments/plot_noise_montage.py --dataset all --coverage
```

The first command writes a montage of the 7 settings and a clean/occluded/mask figure for 8:2. The second writes a table of the mean occluded-pixel fraction instead.

Lambda study (report §4.4). The lambda study uses its own seeds (1000/2000/3000 + r), separate from the seeds of the main experiment. Its raw results are already included in `experiments/lambda_study_results/`, so the selection and its figures can be reproduced without refitting:

```bash
python code/algorithm/experiments/plot_lambda_sensitivity.py
python code/algorithm/experiments/plot_objective_plateau.py
python code/algorithm/experiments/rre_trajectory.py --dataset orl --plot-only
```

To recompute the raw results (this takes a long time):

```bash
python code/algorithm/experiments/rnmf_lambda_sensitivity.py --dataset orl
python code/algorithm/experiments/rnmf_lambda_sensitivity.py --dataset yaleb
python code/algorithm/experiments/rre_trajectory.py --dataset orl
```

- **Selection rule:** on setting 8:2, take the λ with the lowest mean RRE. Every λ whose mean RRE is within one standard deviation (ddof = 1) of that best mean counts as tied, and the largest tied λ is chosen.
- **Result:** ORL λ = 0.01, YaleB λ = 0.03.

Tests:

```bash
python -m pytest -q code/algorithm
```

Tests that need the supplied datasets are skipped when `code/data/` is empty.

## Outputs

Each `main.py` run writes a new folder, `code/results/<YYYYmmdd-HHMMSS>_<commit>_<dataset>/`. Existing folders are never overwritten.

Without Git (for example when the code is run from the submission zip, or when `git` is not installed), `<commit>` is `nogit`, and `metadata.json` records `"commit": null` and `"dirty": null`.

The folder contains:

| File | Content |
|---|---|
| `metadata.json` | the full protocol (λ, tol, max_iter, min_iter for each method), rank, seeds, settings, git commit with a dirty flag, library versions, start/end time and runtime |
| `raw_results.csv` | one row per (setting, repeat, method): RRE, `rre_occluded`, `rre_unoccluded` (NaN when m = 0), ACC, NMI, iterations, `converged`, runtime, objective |
| `summary.csv` | mean and std (ddof = 1) per setting and method, plus the paired difference `paired_rre_diff` = RRE(NMF) − RRE(RNMF-L1) over repeats (positive means RNMF-L1 is better) |
| `objective_trace.csv` | objective after every iteration for repeat 0 (setting, method, iteration, objective) |
| `plots/` | `<dataset>_rre_sweeps` (RRE vs b with m = 1, and RRE vs m with b = 8), `<dataset>_objective`, `<dataset>_summary_table.tex` |
| `reconstructions/` | clean, occluded and reconstructed example images (these show faces; do not commit them) |

The figure scripts write to `code/figures/noise/` and `code/figures/lambda_study/`.

## Runtime

Runtimes were measured on an Apple-silicon laptop (macOS, Python 3.10, NumPy 1.26). A condition is one (setting, repeat) pair: both methods are fitted and KMeans is run.

| | per condition | 1 repeat (7 settings) | 5 repeats (full run) |
|---|---|---|---|
| ORL | ≈ 26 s (NMF ≈ 3 s, RNMF-L1 ≈ 22 s) | 3.1 min (measured) | ≈ 15 min |
| YaleB | ≈ 3.3 min (NMF ≈ 18 s, RNMF-L1 ≈ 175 s) | ≈ 23 min | ≈ 2 h |

- **ORL:** the 1-repeat time was measured.
- **YaleB:** the per-condition time was measured on setting 8:2. The other figures are extrapolations: RNMF-L1 runs a fixed number of iterations, so its time is about the same for every setting.

Most of the time is spent in the fixed 2000 iterations of RNMF-L1.
