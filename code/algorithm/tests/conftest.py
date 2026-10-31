"""Shared pytest configuration for the Member 3 data and noise tests.

The supplied ORL and Extended YaleB datasets are never committed. They live in
code/data, which is located here from this file's own path, so the
tests do not depend on the current working directory. Tests that need the
supplied data request the ``supplied_data`` fixture; they are skipped, not
failed, when either dataset directory is missing.
"""

from pathlib import Path
import sys

import pytest

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ALGORITHM_DIR.parent / "data"
SUPPLIED_DATASETS = ("ORL", "CroppedYaleB")

# Make ``utils``, ``nmf`` and ``experiments`` importable under a plain
# ``pytest`` call from any directory.
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))


@pytest.fixture(scope="session")
def data_root():
    """Return code/data, skipping when a supplied dataset is missing."""
    missing = [name for name in SUPPLIED_DATASETS if not (DATA_ROOT / name).is_dir()]
    if missing:
        pytest.skip(
            f"supplied dataset(s) {', '.join(missing)} not found in {DATA_ROOT}; "
            "place the ORL/ and CroppedYaleB/ folders there to run these checks"
        )
    return DATA_ROOT


@pytest.fixture
def supplied_data(request, data_root):
    """Expose ``data_root`` to unittest-style tests as ``self.data_root``."""
    request.instance.data_root = data_root
