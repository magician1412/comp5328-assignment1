"""Default locations of the data, results and figures folders.

Every path is resolved from this file, never from the current working
directory, so the scripts behave the same wherever they are launched. The
layout mirrors the submission archive::

    code/
    ├── algorithm/   (this package's parent: main.py, nmf/, utils/, ...)
    ├── data/        ORL/ and CroppedYaleB/ (never committed)
    ├── results/     one folder per main.py run (generated)
    └── figures/     generated figures
"""

from pathlib import Path

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ALGORITHM_DIR.parent
DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = CODE_DIR / "results"
FIGURES_DIR = CODE_DIR / "figures"
