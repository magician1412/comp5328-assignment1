# COMP4328/5328 Assignment 1 — NMF robustness under occlusion

Group project, due 8 Oct 2026. Implement NMF algorithms from scratch and compare their robustness when face images are corrupted by contiguous occlusion blocks. Report 80 marks, code 20 marks.

---

## 1. Assignment constraints — must obey

These come from the assignment specification / teaching clarification and take priority over project conventions below.

1. **Implement at least two NMF algorithms, with at least one method not taught in the course.**
   Our locked primary comparison is:
   - Standard Frobenius-Norm NMF (Lee–Seung multiplicative updates) as the taught baseline.
   - RNMF-\(L_1\) from Zhang, Chen, Zheng & He (2011), *Robust non-negative matrix factorization*, as the robust non-taught method.

2. **NMF implementations may use only Python standard library, NumPy and SciPy.**
   Never use `sklearn.decomposition.NMF`, Nimfa, or another library containing an NMF solver.

3. **Scikit-learn is for evaluation only.**
   It may be used for KMeans and evaluation metrics, but never for NMF factorization.

4. **Only occlusion noise is permitted for the assignment experiments.**
   Do not add Gaussian noise, uniform additive noise, salt-and-pepper noise, or other noise families.

5. **Different occlusion parameter settings count as different occlusion-noise types/settings.**
   For example, changing block edge length `b` and/or number of blocks `m` creates different occlusion settings.

6. **Demonstrate the occlusion settings used in the experiments.**
   The report should visually show the clean image and representative contaminated images for the main settings used. Prefer a compact montage rather than a separate large figure for every condition.

7. **RRE is mandatory.**
   For factors learned from contaminated data `V`, evaluate reconstruction against the corresponding clean data `V_hat`:

   `RRE = ||V_hat - W @ H||_F / ||V_hat||_F`

8. **Datasets are never committed.**
   The dataset directory must remain ignored by Git and the submitted `code/data/` directory must be empty. Markers will supply the datasets.

9. **The report must use LaTeX.**
   The required report structure follows a research-paper style. The assignment specification takes priority over formatting defaults in the supplied LaTeX template.

10. **Related Work should connect the project to both conventional and robust NMF literature.**
    Briefly include conventional non-robust NMF for context, then place more emphasis on robust NMF methods relevant to occlusion/outliers. Discuss main ideas, advantages and limitations rather than listing papers. Technical details of our selected algorithms belong in Methodology.

---

## 2. Project conventions — deliberate defaults, not assignment law

These conventions are chosen to make the code auditable, reproducible and easy to integrate across four group members. They may be changed only deliberately and consistently.

1. **Samples are stored as columns.**
   `V` has shape `(d, n)`: each column is one flattened image, `d` = pixels, `n` = images. Do not silently transpose.

2. **Library boundaries by module.**
   Keep third-party dependencies isolated so compliance is easy to audit:
   - `sklearn` → evaluation/metrics code only
   - `PIL` → data loading / image I/O only
   - `matplotlib` → visualization only
   - `numpy`, `scipy` → anywhere

3. **Convert image data to floating point before NMF or corruption.**
   Use a documented intensity scale consistently. Preferred default: normalize to `[0, 1]`, so white occlusion is `1.0`.

4. **Preserve a clean reference matrix.**
   Never overwrite the clean data used for RRE. Corruption is applied to a copy.

5. **Derive dataset counts from disk at runtime.**
   The verified counts below are validation targets, not values to hardcode into the loader.

---

## 3. Verified dataset facts

These were verified by scanning the supplied local datasets. If the loader produces different counts/shapes for this release, investigate the loader rather than hardcoding these values.

| | ORL | Extended YaleB |
|---|---|---|
| Subject folders | 40 (`s1`–`s40`) | 38 (`yaleB01`–`yaleB39`, `yaleB14` absent) |
| Usable images | 400 | 2414 |
| Native size | 92×112 | 168×192 |
| Recommended resize | `reduce=3` → `(1110, 400)` | `reduce=4` → `(2016, 2414)` |

Files the loader must skip:
- ORL: `.DS_Store`, `README` at top level.
- YaleB: `.DS_Store`, `WS_FTP.LOG`, `*.info`, `*.pgm.bad`, `DEADJOE`, and `*Ambient.pgm`.

Notes:
- YaleB has 2452 `.pgm` files in total; excluding 38 Ambient images leaves 2414 usable images.
- The 18 `.pgm.bad` files mean some classes contain 59–63 usable images instead of 64, so YaleB is mildly class-imbalanced.
- The supplied cropped YaleB release used here contains pose token `P00`; do not claim the local experimental subset contains nine poses merely because the general dataset description mentions nine poses.

---

## 4. Fixes to the provided notebook loader

The notebook is example code, not the final implementation. Fix these issues in our loader:

1. **Floating-point conversion.**
   `np.asarray(img)` yields `uint8`. Convert to `float64` (or another explicitly chosen floating dtype) before NMF/noise processing and document the intensity range.

2. **Label construction.**
   Filter to valid subject directories before enumeration so labels are contiguous `0..K-1` and are not shifted by files such as `.DS_Store` or `README`.

3. **Deterministic ordering.**
   Sort subject directories and image filenames explicitly. Filesystem iteration order must not affect column order.

4. **Explicit resize behavior.**
   Specify the resize interpolation method explicitly rather than relying on a Pillow default.

---

## 5. Locked primary algorithms

### 5.1 Baseline: Frobenius-Norm NMF

Objective:

`min_{W>=0,H>=0} ||V - W H||_F^2`

Use Lee–Seung multiplicative updates with a small numerical guard `eps` in denominators.

For a current nonnegative data matrix `X`:

`H <- H * (W.T @ X) / (W.T @ W @ H + eps)`

`W <- W * (X @ H.T) / (W @ H @ H.T + eps)`

Keep non-negativity after updates and track the objective value for convergence diagnostics.

### 5.2 Robust method: RNMF-L1 (Zhang et al., 2011)

Use the sparse-error formulation:

`min_{W>=0,H>=0,S} ||V - W H - S||_F^2 + lambda * ||S||_1`

where `S` is a **signed sparse corruption matrix**. Do not constrain `S >= 0`.

#### Sparse-error update

With `R = V - W @ H`, update `S` by element-wise soft thresholding at `lambda / 2`:

`S = sign(R) * maximum(abs(R) - lambda/2, 0)`

This threshold corresponds to the objective above without a `1/2` coefficient on the squared Frobenius term.

#### W/H updates

After updating `S`, define the current de-corrupted matrix:

`X_tilde = V - S`

Then perform nonnegative multiplicative updates on `X_tilde`:

`W <- W * (X_tilde @ H.T) / (W @ H @ H.T + eps)`

`H <- H * (W.T @ X_tilde) / (W.T @ W @ H + eps)`

Use the algorithm order `S -> W -> H -> optional normalization` unless a verified implementation detail from the original paper requires otherwise.

If columns of `W` are normalized, compensate the corresponding rows of `H` so that `W @ H` is unchanged.

#### Reconstruction used for evaluation

For RNMF-L1, `W @ H` is the recovered clean/low-rank component. Do **not** include `S` in the RRE reconstruction. `S` represents corruption.

#### Hyperparameter lambda

Do not treat one literature value as universally optimal. Preferred protocol:
- normalize pixels to `[0,1]`;
- perform a small, pre-specified lambda-sensitivity study on a moderate occlusion condition;
- choose one fixed lambda per dataset, or one global lambda if justified;
- freeze the chosen lambda before the main robustness sweep.

Do not tune lambda separately for every test condition after seeing the results.

---

## 6. Code organization

The repository mirrors the submission archive: `code/algorithm/` holds all code and `code/data/` is empty in Git (it keeps only `.gitkeep`; the datasets are placed there locally and are gitignored).

```text
Assignment1/
├── code/
│   ├── algorithm/                   # everything submitted as code
│   │   ├── main.py                  # command-line entry point (main experiment)
│   │   ├── nmf/                     # NumPy/SciPy only
│   │   │   ├── nmf_baseline.py      # Frobenius NMF (Lee–Seung)
│   │   │   └── nmf_rnmf_l1.py       # Zhang et al. RNMF-L1
│   │   ├── utils/                   # data_loader, sampling, occlusion,
│   │   │                            # evaluation_metrics, visualize, paths
│   │   ├── experiments/             # experiment runner, plots, lambda study
│   │   ├── tests/                   # pytest suite (incl. conftest.py)
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── data/                        # ORL/, CroppedYaleB/ locally; EMPTY in submission
│   ├── results/                     # generated (one folder per main.py run), gitignored
│   └── figures/                     # generated, gitignored
├── CLAUDE.md, instruction.pdf, assignment1.ipynb, .gitignore
```

Default paths are resolved from `code/algorithm/utils/paths.py` (`__file__`), never from the current directory, so every script runs from any working directory.

Separation of concerns:
- Algorithm modules must not load datasets, add noise, or perform plotting.
- Noise generation must not depend on a specific NMF algorithm.
- Metrics/evaluation must not call a particular NMF implementation internally.
- Visualization should read saved result files rather than rerunning expensive experiments.

A shared base class is optional. Use one only if it does not force mathematically different algorithms into an inappropriate common update loop. Shared utilities for initialization, validation, seeds, stopping checks and history tracking are encouraged where genuinely common.

---

## 7. Shared interfaces

Recommended interfaces:

```python
# data_loader.py
load_data(root, reduce, dtype=np.float64, normalize=True) -> V, labels, img_shape
subsample(V, labels, frac=0.9, seed=0) -> V_sub, labels_sub, indices

# noise.py
add_occlusion(V, img_shape, b, m, seed=0, fill=None, allow_overlap=True) -> V_noisy, mask
corrupted_fraction(mask) -> float

# metrics.py
rre(V_clean, W, H) -> float
```

All NMF implementations should expose a consistent high-level API, for example:

```python
fit(V) -> W, H, history
```

RNMF-L1 may additionally return `S` (for example `fit(V) -> W, H, S, history`) or expose it as fitted state. Do not hide the corruption estimate because it may be useful for diagnostics and report figures.

The shared interface should standardize inputs, outputs, initialization/seed handling and stopping metadata without forcing identical internal update logic.

---

## 8. Occlusion protocol

1. **Only contiguous square occlusion is used.**

2. For each image, generate `m` square blocks of edge length `b` at random valid positions.

3. Choose block coordinates so that the entire `b x b` block lies inside the resized image. Do not clip a nominal block at an image boundary because that changes its actual corrupted area.

4. Fill occluded pixels with the maximum value of the chosen image scale (`1.0` if normalized to `[0,1]`, otherwise `255`).

5. Decide overlap policy explicitly. The default may allow overlap, but the implementation must return the actual pixel mask so the true corrupted fraction can be measured. If overlap is disallowed, implement that deliberately and document it.

6. Apply corruption only to the working copy `V`; preserve the matching clean matrix `V_clean` for RRE.

7. When using 90% subsampling, `V_clean` must be the clean version of the **same sampled columns**, not the full dataset.

8. Use the exact same sampled indices and corruption masks when comparing algorithms within one run/configuration.

### Recommended experiment organization

Prefer interpretable sweeps rather than an unnecessarily large grid:

- **Block-size study:** fix `m` and vary `b`.
- **Block-count study:** fix `b` and vary `m`.
- Optional combined-severity conditions only if they add a clear analysis.

All main occlusion settings used in the quantitative experiments should be visually demonstrated in a compact clean/noisy montage.

---

## 9. Evaluation protocol

1. **Primary metric: RRE** against the clean corresponding subset.

2. **Optional clustering evaluation:** run KMeans on `H.T` with number of clusters equal to number of subjects.

3. **Accuracy:** if following the tutorial convention, map each cluster to its majority ground-truth class before computing accuracy.

4. **NMI:** compute NMI directly between ground-truth labels and raw cluster IDs. NMI is permutation-invariant and does not require majority label mapping.

5. **Repeated evaluation:** use multiple repeats (default: 5) with 90% sampling, as suggested by the assignment. Report mean and standard deviation.

6. Compare algorithms using the same:
   - sampled columns,
   - corruption masks,
   - rank `k`,
   - initialization seed where comparable,
   - convergence/stopping metadata where mathematically appropriate.

7. Track runtime and iteration/objective history where useful for code-quality and discussion sections.

---

## 10. Reproducibility

- Every stochastic project function takes an explicit seed.
- Use `rng = np.random.default_rng(seed)` rather than global `np.random.*` state.
- Seeds must control subsampling, NMF initialization, occlusion placement and KMeans initialization.
- Same inputs and seeds should deterministically reproduce sampling, initialization and corruption masks in the same software environment.
- Do not require bit-identical floating-point results across different machines/BLAS implementations; small numerical differences may occur.
- Record enough metadata to reproduce every plotted/table result: dataset, resize factor, rank, algorithm, lambda (if used), `b`, `m`, repeat index, sample seed, corruption seed, initialization seed, stopping rule and runtime.

---

## 11. Numerical and code-quality rules

- Prefer vectorized NumPy/SciPy operations. Avoid Python loops over individual pixels.
- Loops over images, iterations, configurations or blocks are acceptable where they improve clarity and are not the main computational bottleneck.
- Add docstrings to every public function/class.
- Validate matrix non-negativity before NMF fitting.
- Guard multiplicative-update denominators with `eps` (default around `1e-10`).
- Clamp tiny negative floating-point values to zero where appropriate for nonnegative factors.
- Detect/handle NaN or Inf early and fail with a useful error message.
- Track objective history so convergence behavior can be inspected rather than assumed.
- Code should remain readable and modular; code quality and feasible runtime are directly graded.

---

## 12. Reporting guidance relevant to implementation

### Related Work

Use a broad-to-specific structure:
1. conventional NMF (brief context),
2. why squared/Frobenius reconstruction can be sensitive to large outliers,
3. robust NMF families (for example L1-based, L2,1-based, sparse-error, Cauchy-type approaches),
4. position our selected RNMF-L1 method within that literature.

Discuss ideas, advantages and limitations. Do not duplicate the full derivation that belongs in Methodology.

### Methodology

For each implemented algorithm describe:
- objective function,
- constraints,
- optimization/update steps,
- initialization/stopping rules,
- preprocessing,
- occlusion generation,
- why the method is or is not expected to be robust to sparse large-magnitude occlusion.

### Experiments / Discussion

Do not just report numbers. Analyze how performance changes with block size/count and connect the observations to the objectives of the two algorithms. Treat expected behavior as a hypothesis until the measured results support it.

---

## 13. Things you must not do

- Do not invent, hardcode, or fabricate experimental results.
- Do not claim one method is more robust until the experiments support it.
- Do not use the notebook's example `RRE = 0.2269` as a correctness target; it comes from a different noise/model setup.
- Do not use non-occlusion noise in the assignment experiments.
- Do not use external NMF implementations.
- Do not tune RNMF lambda separately for each final test condition after viewing the answer.
- Do not hardcode dataset counts/labels into the loader instead of deriving them from disk.
- Do not silently change matrix orientation.
- Do not let report prose claim dataset properties that are not true of the supplied local files.
- If an assignment instruction, Ed clarification, verified paper formulation, or explicit user request conflicts with a project convention here, stop and flag the conflict instead of guessing.

---

## 14. Primary references to keep consistent

- Lee, D. D., & Seung, H. S. (2001). *Algorithms for non-negative matrix factorization.* Advances in Neural Information Processing Systems.
- Zhang, L., Chen, Z., Zheng, M., & He, X. (2011). *Robust non-negative matrix factorization.*

Do not mix RNMF-L1 update equations from Zhang et al. (2011) with similarly named but mathematically different robust/L1-NMF papers.
