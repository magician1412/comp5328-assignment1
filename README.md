# Robustness of NMF under Block Occlusion

Group project for COMP5328 Advanced Machine Learning (University of Sydney,
2026, four members). We implement two non-negative matrix factorisation (NMF)
algorithms from scratch in NumPy and compare how well they recover clean face
images when the input is corrupted by white square occlusion blocks.

- **Frobenius NMF**: `min ‖V − WH‖²_F`, Lee–Seung multiplicative updates.
- **RNMF-L1** (Zhang et al., 2011): `min ‖V − WH − S‖²_F + λ‖S‖₁`, where the
  signed sparse matrix `S` absorbs the occlusion and `WH` is the recovered
  clean part.

Datasets: ORL (40 subjects, 400 images) and the cropped Extended YaleB
(38 subjects, 2414 images). The main metric is the relative reconstruction
error RRE = ‖V̂ − WH‖_F / ‖V̂‖_F against the clean images V̂.

## Results

Mean RRE over five repeats (90% subsets) for each occlusion setting
(`b` = block edge in pixels, `m` = number of blocks). Lower is better.

| b, m | ORL: Frobenius NMF | ORL: RNMF-L1 | YaleB: Frobenius NMF | YaleB: RNMF-L1 |
|---|---|---|---|---|
| no occlusion | **0.128** | 0.184 | **0.187** | 0.209 |
| 4, 1 | **0.160** | 0.185 | **0.203** | 0.209 |
| 8, 1 | 0.273 | **0.197** | 0.358 | **0.214** |
| 12, 1 | 0.385 | **0.300** | 0.535 | **0.461** |
| 16, 1 | 0.506 | **0.472** | 0.714 | **0.678** |
| 8, 2 | 0.361 | **0.222** | 0.470 | **0.236** |
| 8, 3 | 0.423 | **0.261** | 0.557 | **0.311** |

RNMF-L1 is more accurate once the occlusion is non-trivial; on clean or lightly
occluded data the sparse term also absorbs genuine detail and Frobenius NMF is
slightly better. Raw results, plots and run metadata are in `final_results/`.

## Protocol in brief

- Rank `k` = number of subjects (ORL 40, YaleB 38); both methods share the same
  subset, occlusion mask and initial factors in every repeat.
- Frobenius NMF stops by a relative-change tolerance (window 5, min 200,
  max 2000 iterations).
- RNMF-L1 runs a fixed budget of 2000 iterations with λ = 0.01 (ORL) and
  λ = 0.03 (YaleB). A tolerance rule stops RNMF-L1 on an early objective
  plateau, and its error against the clean data first falls and then rises
  as `WH` slowly learns the blocks, so λ was chosen at a fixed budget on a
  separate tuning setting with seeds disjoint from the evaluation.
  The lambda study and convergence analysis are in
  `code/algorithm/experiments/` and `code/algorithm/experiments/lambda_study_results/`.

## My contribution

I was responsible for RNMF-L1: the implementation (`code/algorithm/nmf/nmf_rnmf_l1.py`)
and its unit tests, the diagnosis of its convergence behaviour (false
convergence on the objective plateau, RRE trajectories, effect of the rank),
the λ sensitivity study and the resulting stopping protocol, and the report
sections on robust NMF, the RNMF-L1 method, the hyperparameters and the
sensitivity and convergence analysis.

## Running the code

See [`code/algorithm/README.md`](code/algorithm/README.md) for setup, all
commands and expected runtimes. In short:

```bash
pip install -r code/algorithm/requirements.txt
python code/algorithm/main.py --dataset orl --repeats 1   # quick check
python -m pytest -q code/algorithm                       # test suite
```

## Not included

- **Datasets.** They are not redistributed here. Download them from the
  original sources and place them in `code/data/ORL` and
  `code/data/CroppedYaleB`:
  - ORL (AT&T Database of Faces): https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  - Extended Yale Face Database B (cropped): https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b
- **Course materials** (assignment specification, tutorial notebook and report
  template) and the report itself.
