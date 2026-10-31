"""Provenance helpers for experiment runs: git state, versions, run folders.

Nothing here affects the numerical results; it only records how a run was
produced. Git is optional. When the git binary is missing, or this code is
not tracked by a Git checkout (for example when it is run from the unzipped
submission archive, even inside some unrelated repository), the commit and
the dirty flag are recorded as None (null in metadata.json) and run folders
use the label "nogit".
"""

from datetime import datetime
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys

from utils.paths import ALGORITHM_DIR

# Distribution names as installed by pip (requirements.txt).
PACKAGES = ("numpy", "scipy", "scikit-learn", "matplotlib", "pillow")
# Run-folder label used instead of a commit hash when Git is unavailable.
NO_GIT_LABEL = "nogit"


def _git(*args):
    """Return the stdout of a git command run in this checkout, or None.

    None is returned when git is not installed (FileNotFoundError and other
    OSErrors), when the command fails (e.g. not a repository) or hangs.
    """
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ALGORITHM_DIR, capture_output=True, text=True,
            check=True, timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def git_state():
    """Return {"commit", "commit_short", "dirty"} for the current checkout.

    ``dirty`` is True when tracked files differ from HEAD (staged or not);
    untracked files are ignored because generated outputs are gitignored.
    All three values are None when git is missing or main.py is not tracked
    by a repository (so an enclosing, unrelated repository is not reported).
    """
    no_git = {"commit": None, "commit_short": None, "dirty": None}
    if _git("ls-files", "--error-unmatch", "main.py") is None:
        return no_git
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return no_git
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(status),
    }


def environment_versions():
    """Return the Python, platform and package versions used for a run."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def make_run_dir(out_dir, started, commit_short, dataset):
    """Create and return a new ``<YYYYmmdd-HHMMSS>_<commit>_<dataset>`` folder.

    ``commit_short`` None (no Git) gives the label "nogit". An existing
    folder is never reused: if the name is taken (two runs in the same
    second), a numeric suffix is appended.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = commit_short or NO_GIT_LABEL
    stem = f"{started:%Y%m%d-%H%M%S}_{label}_{dataset}"
    candidate, suffix = out_dir / stem, 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = out_dir / f"{stem}_{suffix}"


def timestamp(moment=None):
    """Return an ISO-8601 local timestamp (seconds precision)."""
    return (moment or datetime.now()).isoformat(timespec="seconds")


def command_line():
    """Return the command line that started this process."""
    return " ".join([Path(sys.executable).name, *sys.argv])
